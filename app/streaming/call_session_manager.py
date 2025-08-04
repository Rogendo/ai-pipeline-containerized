# app/streaming/call_session_manager.py
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import asyncio

from ..config.settings import redis_task_client
from .progressive_processor import progressive_processor

# Import agent notification service
try:
    from ..services.agent_notification_service import agent_notification_service
    AGENT_NOTIFICATIONS_ENABLED = True
except ImportError:
    AGENT_NOTIFICATIONS_ENABLED = False
    logger.warning("Agent notification service not available")

logger = logging.getLogger(__name__)

@dataclass
class CallSession:
    """Represents an active call session"""
    call_id: str
    start_time: datetime
    last_activity: datetime
    connection_info: Dict
    transcript_segments: List[Dict]
    cumulative_transcript: str
    total_audio_duration: float
    segment_count: int
    status: str  # 'active', 'completed', 'timeout'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CallSession':
        """Create from dictionary"""
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        return cls(**data)

class CallSessionManager:
    """Manages multiple simultaneous call sessions with cumulative transcription"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis_task_client
        self.active_sessions: Dict[str, CallSession] = {}
        self.session_timeout = timedelta(minutes=30)  # Timeout inactive sessions
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self._cleanup_task = None
        
    async def start_session(self, call_id: str, connection_info: Dict) -> CallSession:
        """Start a new call session"""
        try:
            now = datetime.now()
            
            session = CallSession(
                call_id=call_id,
                start_time=now,
                last_activity=now,
                connection_info=connection_info,
                transcript_segments=[],
                cumulative_transcript="",
                total_audio_duration=0.0,
                segment_count=0,
                status='active'
            )
            
            # Store in memory
            self.active_sessions[call_id] = session
            
            # Store in Redis for persistence
            self._store_session_in_redis(session)
            
            logger.info(f"📞 [session] Started call session: {call_id}")
            logger.info(f"📞 [session] Active sessions: {len(self.active_sessions)}")
            
            # Send call start notification to agent
            if AGENT_NOTIFICATIONS_ENABLED:
                try:
                    await agent_notification_service.send_call_start(call_id, connection_info)
                except Exception as e:
                    logger.error(f"❌ Failed to send call start notification for {call_id}: {e}")
            
            # Start cleanup task if not running
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to start session {call_id}: {e}")
            raise
    
    async def add_transcription(self, call_id: str, transcript: str, 
                              audio_duration: float, metadata: Dict = None) -> Optional[CallSession]:
        """Add transcription segment to call session"""
        try:
            session = await self.get_session(call_id)
            if not session:
                logger.warning(f"⚠️ [session] No active session found for call {call_id}")
                return None
            
            now = datetime.now()
            
            # Create transcript segment
            segment = {
                'segment_id': session.segment_count + 1,
                'timestamp': now.isoformat(),
                'transcript': transcript.strip(),
                'audio_duration': audio_duration,
                'metadata': metadata or {}
            }
            
            # Update session
            session.transcript_segments.append(segment)
            session.segment_count += 1
            session.total_audio_duration += audio_duration
            session.last_activity = now
            
            # Update cumulative transcript with smart concatenation
            session.cumulative_transcript = self._concatenate_transcript(
                session.cumulative_transcript, 
                transcript.strip()
            )
            
            # Trigger progressive processing (translation, NER, classification)
            try:
                processed_window = await progressive_processor.process_if_ready(
                    call_id, 
                    session.cumulative_transcript
                )
                
                if processed_window:
                    logger.info(f"🧠 [session] Progressive processing completed window {processed_window.window_id} for call {call_id}")
                    
                    # Add processing info to segment metadata
                    segment['metadata']['progressive_window'] = processed_window.window_id
                    segment['metadata']['window_processed'] = True
                
            except Exception as e:
                logger.error(f"❌ Progressive processing failed for call {call_id}: {e}")
            
            # Store updated session
            self._store_session_in_redis(session)
            
            # Send transcript segment notification to agent
            if AGENT_NOTIFICATIONS_ENABLED:
                try:
                    await agent_notification_service.send_transcript_segment(
                        call_id, 
                        segment, 
                        session.cumulative_transcript
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to send transcript segment notification for {call_id}: {e}")
            
            logger.info(f"📝 [session] Added segment {segment['segment_id']} to call {call_id}")
            logger.info(f"📝 [session] Transcript length: {len(session.cumulative_transcript)} chars")
            logger.info(f"📝 [session] Total duration: {session.total_audio_duration:.1f}s")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to add transcription to session {call_id}: {e}")
            return None
    
    def _concatenate_transcript(self, existing: str, new_text: str) -> str:
        """Smart concatenation of transcript segments"""
        if not existing:
            return new_text
        
        if not new_text:
            return existing
        
        # Remove redundant repetitions at boundaries
        existing_words = existing.lower().split()
        new_words = new_text.lower().split()
        
        # Check for overlap (last few words of existing match first few words of new)
        max_overlap = min(5, len(existing_words), len(new_words))
        overlap_found = 0
        
        for i in range(1, max_overlap + 1):
            if existing_words[-i:] == new_words[:i]:
                overlap_found = i
        
        if overlap_found > 0:
            # Remove overlapping words from new text
            final_new_words = new_text.split()[overlap_found:]
            if final_new_words:
                return existing + " " + " ".join(final_new_words)
            else:
                return existing
        else:
            # No overlap, simple concatenation
            return existing + " " + new_text
    
    async def get_session(self, call_id: str) -> Optional[CallSession]:
        """Get active session by call ID"""
        # Try Redis first for cross-process compatibility
        try:
            session_data = self._get_session_from_redis(call_id)
            if session_data:
                session = CallSession.from_dict(session_data)
                # Update in-memory cache
                self.active_sessions[call_id] = session
                return session
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session {call_id} from Redis: {e}")
        
        # Fallback to memory (for same-process access)
        if call_id in self.active_sessions:
            return self.active_sessions[call_id]
        
        return None
    
    async def end_session(self, call_id: str, reason: str = "completed") -> Optional[CallSession]:
        """End call session and prepare for AI pipeline processing"""
        try:
            session = await self.get_session(call_id)
            if not session:
                logger.warning(f"⚠️ [session] No session found to end: {call_id}")
                return None
            
            # Update session status
            session.status = reason
            session.last_activity = datetime.now()
            
            # Store final session state
            self._store_session_in_redis(session)
            
            # Finalize progressive processing and trigger summarization
            try:
                final_analysis = await progressive_processor.finalize_call_analysis(call_id)
                if final_analysis:
                    logger.info(f"📋 [session] Progressive analysis finalized for call {call_id}")
                    logger.info(f"📊 [session] Analysis summary: {final_analysis['total_windows_processed']} windows, "
                               f"{final_analysis['final_translation_length']} chars translated")
            except Exception as e:
                logger.error(f"❌ Failed to finalize progressive analysis for call {call_id}: {e}")
            
            # Trigger AI pipeline processing if transcript is substantial
            if len(session.cumulative_transcript.strip()) > 50:  # Minimum threshold
                await self._trigger_ai_pipeline(session)
            
            # Send call end notification to agent
            if AGENT_NOTIFICATIONS_ENABLED:
                try:
                    final_stats = {
                        'duration': session.total_audio_duration,
                        'segments': session.segment_count,
                        'transcript_length': len(session.cumulative_transcript),
                        'start_time': session.start_time.isoformat(),
                        'end_time': session.last_activity.isoformat()
                    }
                    await agent_notification_service.send_call_end(call_id, reason, final_stats)
                except Exception as e:
                    logger.error(f"❌ Failed to send call end notification for {call_id}: {e}")
            
            # Remove from active sessions
            if call_id in self.active_sessions:
                del self.active_sessions[call_id]
            
            logger.info(f"📞 [session] Ended call session: {call_id} (reason: {reason})")
            logger.info(f"📊 [session] Final stats - Duration: {session.total_audio_duration:.1f}s, "
                       f"Segments: {session.segment_count}, "
                       f"Transcript: {len(session.cumulative_transcript)} chars")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to end session {call_id}: {e}")
            return None
    
    async def _trigger_ai_pipeline(self, session: CallSession):
        """Trigger full AI pipeline processing for completed call"""
        try:
            from ..tasks.audio_tasks import process_audio_task
            
            # Create synthetic audio filename for the complete call
            filename = f"call_{session.call_id}_{session.start_time.strftime('%Y%m%d_%H%M%S')}.transcript"
            
            # Convert transcript to bytes (simulate audio processing)
            transcript_bytes = session.cumulative_transcript.encode('utf-8')
            
            # Submit to full AI pipeline
            task = process_audio_task.delay(
                audio_bytes=transcript_bytes,
                filename=filename,
                language="sw",  # Could be stored in session metadata
                include_translation=True,
                include_insights=True
            )
            
            # Store task reference in session metadata
            session_key = f"call_session:{session.call_id}"
            pipeline_info = {
                'task_id': task.id,
                'submitted_at': datetime.now().isoformat(),
                'status': 'processing'
            }
            
            if self.redis_client:
                self.redis_client.hset(
                    session_key, 
                    'ai_pipeline', 
                    json.dumps(pipeline_info)
                )
            
            logger.info(f"🤖 [session] Triggered AI pipeline for call {session.call_id}, task: {task.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger AI pipeline for session {session.call_id}: {e}")
    
    async def get_all_active_sessions(self) -> List[CallSession]:
        """Get all active sessions"""
        return list(self.active_sessions.values())
    
    async def get_session_stats(self) -> Dict:
        """Get statistics about all sessions"""
        active_count = len(self.active_sessions)
        total_duration = sum(s.total_audio_duration for s in self.active_sessions.values())
        total_segments = sum(s.segment_count for s in self.active_sessions.values())
        
        return {
            'active_sessions': active_count,
            'total_audio_duration': total_duration,
            'total_segments': total_segments,
            'average_duration_per_session': total_duration / active_count if active_count > 0 else 0,
            'session_list': [s.call_id for s in self.active_sessions.values()]
        }
    
    def _store_session_in_redis(self, session: CallSession):
        """Store session in Redis for persistence"""
        # Ensure we have a Redis client
        if not self.redis_client:
            from ..config.settings import redis_task_client
            self.redis_client = redis_task_client
            
        if not self.redis_client:
            logger.warning(f"🔍 [session] Redis client not available for storing session {session.call_id}")
            return
        
        try:
            session_key = f"call_session:{session.call_id}"
            session_data = session.to_dict()
            
            # Store main session data
            self.redis_client.hset(session_key, 'data', json.dumps(session_data))
            
            # Set expiration (keep for 24 hours after last activity)
            expire_time = int((session.last_activity + timedelta(hours=24)).timestamp())
            self.redis_client.expireat(session_key, expire_time)
            
            # Add to active sessions set
            if session.status == 'active':
                self.redis_client.sadd('active_call_sessions', session.call_id)
            else:
                self.redis_client.srem('active_call_sessions', session.call_id)
                
            logger.debug(f"🔍 [session] Successfully stored session {session.call_id} in Redis")
                
        except Exception as e:
            logger.error(f"❌ Failed to store session {session.call_id} in Redis: {e}")
    
    def _get_session_from_redis(self, call_id: str) -> Optional[Dict]:
        """Retrieve session from Redis"""
        # Ensure we have a Redis client
        if not self.redis_client:
            from ..config.settings import redis_task_client
            self.redis_client = redis_task_client
            
        if not self.redis_client:
            logger.warning(f"🔍 [session] Redis client not available for session {call_id}")
            return None
        
        try:
            session_key = f"call_session:{call_id}"
            session_json = self.redis_client.hget(session_key, 'data')
            
            if session_json:
                logger.debug(f"🔍 [session] Found session {call_id} in Redis")
                return json.loads(session_json)
            else:
                logger.debug(f"🔍 [session] No Redis data found for session {call_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session {call_id} from Redis: {e}")
        
        return None
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of inactive sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                logger.info("📞 [session] Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
    
    async def _cleanup_inactive_sessions(self):
        """Clean up sessions that have been inactive too long"""
        now = datetime.now()
        timeout_threshold = now - self.session_timeout
        
        inactive_sessions = []
        
        for call_id, session in list(self.active_sessions.items()):
            if session.last_activity < timeout_threshold:
                inactive_sessions.append(call_id)
        
        for call_id in inactive_sessions:
            logger.info(f"🧹 [session] Cleaning up inactive session: {call_id}")
            await self.end_session(call_id, reason="timeout")
        
        if inactive_sessions:
            logger.info(f"🧹 [session] Cleaned up {len(inactive_sessions)} inactive sessions")

# Global session manager instance
call_session_manager = CallSessionManager()