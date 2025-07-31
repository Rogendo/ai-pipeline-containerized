# app/streaming/tcp_server.py - Updated to use Celery
import asyncio
import logging
import numpy as np
from typing import Dict
from datetime import datetime

from .audio_buffer import AsteriskAudioBuffer
from ..tasks.audio_tasks import process_streaming_audio_task  # Use your existing Celery tasks

logger = logging.getLogger(__name__)

class AsteriskTCPServer:
    """TCP server for Asterisk audio input - uses Celery workers"""
    
    def __init__(self, model_loader=None):
        # Don't need model_loader anymore - we'll use Celery
        self.active_connections: Dict[str, AsteriskAudioBuffer] = {}
        self.server = None
        
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle Asterisk connection"""
        client_addr = writer.get_extra_info('peername')
        connection_id = f"{client_addr[0]}:{client_addr[1]}:{datetime.now().strftime('%H%M%S')}"
        
        logger.info(f"🎙️ [client] Connection from {client_addr} → {connection_id}")
        
        # Create buffer for this connection
        audio_buffer = AsteriskAudioBuffer()
        self.active_connections[connection_id] = audio_buffer
        
        # Protocol state
        uid_buffer = bytearray()
        audio_mode = False
        
        try:
            while True:
                # Receive 20ms SLIN (640 bytes)
                data = await reader.read(640)
                
                if not data:
                    logger.info(f"🔌 [client] Connection closed by {client_addr}")
                    break
                
                if not audio_mode:
                    # Handle UID protocol
                    for byte in data:
                        if byte == 13:  # CR
                            uid_str = uid_buffer.decode('utf-8', errors='ignore')
                            logger.info(f"🆔 [client] uid={uid_str}")
                            audio_mode = True
                            break
                        uid_buffer.append(byte)
                    continue
                
                # Process audio data
                audio_array = audio_buffer.add_chunk(data)
                
                if audio_array is not None:
                    # Submit to Celery for transcription
                    await self._submit_transcription(audio_array, connection_id)
                    
        except Exception as e:
            logger.error(f"❌ [client] Error handling connection {connection_id}: {e}")
        finally:
            # Cleanup
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            writer.close()
            await writer.wait_closed()
            logger.info(f"🧹 Cleaned up connection {connection_id}")
            
    async def _submit_transcription(self, audio_array: np.ndarray, connection_id: str):
        """Submit transcription to Celery worker"""
        try:
            # Convert numpy array to bytes for Celery
            audio_bytes = (audio_array * 32768.0).astype(np.int16).tobytes()
            
            # Create synthetic filename
            timestamp = datetime.now().strftime("%H%M%S%f")[:-3]  # milliseconds
            filename = f"stream_{connection_id}_{timestamp}.wav"
            
            # Submit to your existing Celery task
            task = process_streaming_audio_task.delay(
                audio_bytes=audio_bytes,
                filename=filename,
                connection_id=connection_id,
                language="sw",
                sample_rate=16000,
                duration_seconds=5.0,
                is_streaming=True
            )
            
            logger.info(f"🎵 Submitted transcription task {task.id} for {connection_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to submit transcription for {connection_id}: {e}")
            
    async def start_server(self, host: str = "0.0.0.0", port: int = 8300):
        """Start TCP server"""
        try:
            self.server = await asyncio.start_server(
                self.handle_connection, 
                host, 
                port
            )
            
            logger.info(f"🚀 [Main] Asterisk TCP server listening on {host}:{port}")
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            logger.error(f"❌ Failed to start TCP server: {e}")
            raise
            
    async def stop_server(self):
        """Stop server gracefully"""
        if self.server:
            logger.info("🛑 Stopping Asterisk TCP server...")
            self.server.close()
            await self.server.wait_closed()
            logger.info("✅ Asterisk TCP server stopped")
            
    def get_status(self) -> dict:
        """Get server status"""
        return {
            "server_running": self.server is not None,
            "active_connections": len(self.active_connections),
            "connections": {
                conn_id: buffer.get_stats() 
                for conn_id, buffer in self.active_connections.items()
            },
            "transcription_method": "celery_workers",
            "tcp_port": 8300
        }