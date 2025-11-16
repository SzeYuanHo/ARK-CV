#!/usr/bin/env python3
"""
Gradio UI for Agentic RAG with Knowledge Graph.

This provides a web-based interface with image upload support.
"""

import os
import asyncio
import aiohttp
import gradio as gr
from typing import List, Tuple, Optional, Dict, Any
import base64
from pathlib import Path
import json

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8058")


class GradioRAGInterface:
    """Gradio interface for the Agentic RAG system."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
        self.user_id = "gradio_user"
        # Cache for image characteristics to avoid re-running VLM inference
        self.cached_image_characteristics = None
        self.cached_image_analysis_text = None
        # Track the last processed image path to detect new uploads
        self.last_image_path = None
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using the VLM through the backend."""
        try:
            # Import here to avoid issues if not installed
            from agent.tools import analyze_image_for_cv_features
            
            # Analyze the image
            analysis = await analyze_image_for_cv_features(image_path)
            return analysis
        except Exception as e:
            return {
                "raw_analysis": f"Error analyzing image: {str(e)}",
                "image_characteristics": [f"Error: {str(e)}"]
            }
    
    async def chat_with_agent(
        self,
        message: str,
        history: List[Tuple[str, str]],
        image: Optional[str] = None
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """Send message to agent and return updated history."""
        image_analysis_text = ""
        tools_used_text = ""
        full_message = message  # Default to original message
        
        # Check if this is a new image or the same one from previous request
        is_new_image = image is not None and image != self.last_image_path
        
        # Debug logging
        if image is not None:
            if is_new_image:
                print(f"🆕 New image detected: {image}")
            else:
                print(f"🔄 Same image as before: {image}")
        else:
            print(f"No image provided")
        
        # Handle image analysis - only run VLM if NEW image provided
        if is_new_image:
            # New image uploaded - run VLM inference and cache results
            print(f"Analyzing new image with VLM...")
            self.last_image_path = image  # Track this image
            try:
                image_features = await self.analyze_image(image)
                image_analysis_text = image_features.get("raw_analysis", "No analysis available")
                characteristics = image_features.get("image_characteristics", [])
                
                # Cache the characteristics for future questions
                self.cached_image_characteristics = characteristics
                self.cached_image_analysis_text = image_analysis_text
                print(f"✅ Image analysis complete. Cached {len(characteristics)} characteristics.")
                
                if characteristics:
                    char_list = "\n".join([f"{char}" for char in characteristics])
                    full_message = f"""Image characteristics from uploaded image:
{char_list}

User question: {message}"""
                else:
                    full_message = f"Image Analysis: {image_analysis_text}\n\nUser question: {message}"
            except Exception as e:
                image_analysis_text = f"Failed to analyze image: {str(e)}"
                full_message = message
        
        elif self.cached_image_characteristics:
            # No new image, but we have cached characteristics from previous upload
            # Reuse them for follow-up questions
            print(f"♻️ Reusing cached image characteristics ({len(self.cached_image_characteristics)} items)")
            image_analysis_text = f"[Using cached analysis]\n{self.cached_image_analysis_text or ''}"
            char_list = "\n".join([f"{char}" for char in self.cached_image_characteristics])
            full_message = f"""Image characteristics from uploaded image:
{char_list}

User question: {message}"""

        # Prepare request data
        request_data = {
            "message": full_message,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "search_type": "hybrid"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/stream",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"API Error ({response.status}): {error_text}"
                        history.append((message, error_msg))
                        yield history, image_analysis_text, ""
                        return

                    # Process streaming response
                    assistant_reply = ""
                    tools_used = []
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                
                                if data.get('type') == 'session':
                                    self.session_id = data.get('session_id')
                                
                                elif data.get('type') == 'text':
                                    content = data.get('content', '')
                                    assistant_reply += content
                                    # Sanitize to remove chunked encoding artifacts
                                    clean_reply = assistant_reply
                                    # For streaming updates in Gradio
                                    yield history + [(message, clean_reply)], image_analysis_text, tools_used_text
                                
                                elif data.get('type') == 'tools':
                                    tools_used = data.get('tools', [])
                                    tools_used_text = self._format_tools_used(tools_used)
                                    
                            except json.JSONDecodeError:
                                continue

                    # Final response - sanitize before displaying
                    clean_reply = assistant_reply
                    if tools_used_text:
                        assistant_reply_with_tools = f"{clean_reply}\n\n{tools_used_text}"
                    else:
                        assistant_reply_with_tools = clean_reply

                    history.append((message, assistant_reply_with_tools))
                    yield history, image_analysis_text, tools_used_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            yield history, image_analysis_text, ""
    

    
    def _format_tools_used(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools used for display."""
        if not tools:
            return ""
        
        formatted = "🛠️ **Tools Used:**\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('tool_name', 'unknown')
            args = tool.get('args', {})
            
            formatted += f"{i}. **{tool_name}**"
            
            # Show key arguments for context
            if args:
                key_args = []
                if 'query' in args:
                    query_text = args['query'][:50] + '...' if len(args['query']) > 50 else args['query']
                    key_args.append(f"query='{query_text}'")
                if 'limit' in args:
                    key_args.append(f"limit={args['limit']}")
                if 'entity_name' in args:
                    key_args.append(f"entity='{args['entity_name']}'")
                
                if key_args:
                    formatted += f" ({', '.join(key_args)})"
            
            formatted += "\n"
        
        return formatted.strip()
    
    def clear_session(self):
        """Clear the current session and cached image characteristics."""
        self.session_id = None
        self.cached_image_characteristics = None
        self.cached_image_analysis_text = None
        self.last_image_path = None
        return [], "", ""
    
    async def check_health(self) -> str:
        """Check API health status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get('status', 'unknown')
                        return f"✅ API Status: {status}"
                    else:
                        return f"⚠️ API returned status code: {response.status}"
        except Exception as e:
            return f"❌ Cannot connect to API: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface."""
    rag_interface = GradioRAGInterface()
    
    # Sync wrapper for async functions
    def sync_chat(message, history, image):
        if not message.strip():
            return history, "", "", ""
        
        async def process_generator():
            result = None
            async for response in rag_interface.chat_with_agent(message, history, image):
                result = response  # Keep updating with latest response
            return result if result else (history, "", "", "")
        
        result = asyncio.run(process_generator())
        # Ensure we always return 4 values: chatbot, msg (empty), image_analysis, tools
        if len(result) == 3:
            return result[0], "", result[1], result[2]
        return result
    
    def sync_health():
        return asyncio.run(rag_interface.check_health())

    # Create the Gradio UI
    with gr.Blocks(title="Agentic RAG with Knowledge Graph", theme=gr.themes.Ocean()) as demo:
        gr.Markdown(
            """
            # Agentic RAG with Knowledge Graph for Microscopy Research
            
            Ask questions about CV architectures in microscopic research. 
            Upload an image for VLM analysis to get personalised architecture recommendations!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask about CV architectures... (For best results, include descriptions of your task: Model Size, Desired Precision, Dataset Size)",
                        lines=2,
                        scale=4
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    health_btn = gr.Button("Check API Status")
                
                health_status = gr.Textbox(
                    label="API Health Status",
                    interactive=False,
                    visible=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Optional: Upload Image")
                image_input = gr.Image(
                    label="Upload microscopy or CV image",
                    type="filepath",
                    height=300
                )
                image_analysis_output = gr.Textbox(
                    label="Image Analysis",
                    lines=6,
                    interactive=False,
                    placeholder="Image analysis will appear here..."
                )
                
                tools_output = gr.Markdown(
                    label="Tools Used",
                    value="",
                    visible=True
                )
                
                gr.Markdown(
                    """
                    ### Tips
                    - Upload an image for VLM-based feature extraction
                    - Follow-up questions will reuse the same image characteristics
                    - Click **Clear Chat** to reset and analyze a new image
                    - Ask about specific papers or architectures
                    - Compare different CV approaches
                    - Request information about research trends
                    - Please be patient, the model takes about 60 seconds to run inference
                    """
                )
        
        # Event handlers
        def submit_message(message, history, image):
            if not message.strip():
                return history, "", image_analysis_output.value, tools_output.value
            
            # Create generator for streaming updates
            async def response_generator():
                async for result in sync_chat(message, history, image):
                    yield result
            
            # Use Gradio's streaming capability
            return gr.update(value=response_generator()), "", "", ""

        # Update the submit click handler
        submit.click(
            fn=sync_chat,
            inputs=[msg, chatbot, image_input],
            outputs=[chatbot, msg, image_analysis_output, tools_output],
            queue=True
        )

        msg.submit(
            fn=sync_chat,
            inputs=[msg, chatbot, image_input],
            outputs=[chatbot, msg, image_analysis_output, tools_output],
            queue=True
        )
        
        clear.click(
            fn=lambda: rag_interface.clear_session(),
            outputs=[chatbot, image_analysis_output, tools_output]
        )
        
        def show_health():
            status = sync_health()
            return gr.update(value=status, visible=True)
        
        health_btn.click(
            fn=show_health,
            outputs=[health_status]
        )
    
    return demo


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio UI for Agentic RAG")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8058",
        help="Base URL for the API"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio UI"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    
    args = parser.parse_args()
    
    # Update global API URL
    global API_BASE_URL
    API_BASE_URL = args.api_url
    
    print(f"🚀 Starting Gradio UI...")
    print(f"📡 API URL: {API_BASE_URL}")
    print(f"🌐 UI will be available at: http://localhost:{args.port}")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
