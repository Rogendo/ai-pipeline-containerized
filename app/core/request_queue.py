import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RequestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class QueuedRequest:
    def __init__(self, request_id: str, request_type: str, priority: int = 5):
        self.request_id = request_id
        self.request_type = request_type
        self.priority = priority
        self.status = RequestStatus.QUEUED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result: Optional[Dict] = None

class RequestQueue:
    """Manages request queuing and status tracking"""
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.requests: Dict[str, QueuedRequest] = {}
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processing_requests: Dict[str, QueuedRequest] = {}
        
        logger.info(f"Request Queue initialized with max_size={max_size}")
    
    async def add_request(self, request_type: str, priority: int = 5) -> str:
        """Add a new request to the queue"""
        if self.queue.full():
            raise RuntimeError("Queue is full. Please try again later.")
        
        request_id = str(uuid.uuid4())
        queued_request = QueuedRequest(request_id, request_type, priority)
        
        self.requests[request_id] = queued_request
        await self.queue.put(queued_request)
        
        logger.info(f"Added request {request_id} to queue (type: {request_type})")
        return request_id
    
    async def get_next_request(self) -> Optional[QueuedRequest]:
        """Get the next request from the queue"""
        try:
            request = await self.queue.get()
            request.status = RequestStatus.PROCESSING
            request.started_at = datetime.now()
            self.processing_requests[request.request_id] = request
            
            logger.info(f"Started processing request {request.request_id}")
            return request
            
        except asyncio.QueueEmpty:
            return None
    
    def complete_request(self, request_id: str, result: Dict = None, error: str = None):
        """Mark a request as completed"""
        if request_id in self.processing_requests:
            request = self.processing_requests[request_id]
            request.completed_at = datetime.now()
            
            if error:
                request.status = RequestStatus.FAILED
                request.error_message = error
            else:
                request.status = RequestStatus.COMPLETED
                request.result = result
            
            del self.processing_requests[request_id]
            logger.info(f"Completed request {request_id} with status {request.status}")
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        if request_id not in self.requests:
            return None
        
        request = self.requests[request_id]
        
        status_info = {
            "request_id": request_id,
            "status": request.status,
            "request_type": request.request_type,
            "created_at": request.created_at.isoformat(),
            "queue_position": self._get_queue_position(request_id) if request.status == RequestStatus.QUEUED else None
        }
        
        if request.started_at:
            status_info["started_at"] = request.started_at.isoformat()
        
        if request.completed_at:
            status_info["completed_at"] = request.completed_at.isoformat()
            status_info["processing_time"] = (request.completed_at - request.started_at).total_seconds()
        
        if request.error_message:
            status_info["error"] = request.error_message
        
        if request.result:
            status_info["result"] = request.result
        
        return status_info
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        return {
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.max_size,
            "processing_count": len(self.processing_requests),
            "total_requests": len(self.requests),
            "completed_requests": len([r for r in self.requests.values() if r.status == RequestStatus.COMPLETED]),
            "failed_requests": len([r for r in self.requests.values() if r.status == RequestStatus.FAILED])
        }
    
    def _get_queue_position(self, request_id: str) -> int:
        """Get position of request in queue (1-based)"""
        # This is an approximation since asyncio.Queue doesn't expose position
        if request_id in self.requests:
            request = self.requests[request_id]
            if request.status == RequestStatus.QUEUED:
                # Count requests created before this one that are still queued
                position = 1
                for other_request in self.requests.values():
                    if (other_request.status == RequestStatus.QUEUED and 
                        other_request.created_at < request.created_at):
                        position += 1
                return position
        return 0

# Global queue instance
request_queue = RequestQueue(max_size=20)
