"""Observability and logging for the RAG application."""

import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, asdict

from config import Config

@dataclass
class LogEntry:
    """Log entry structure."""
    timestamp: str
    level: str
    component: str
    message: str
    metadata: Dict[str, Any]
    query_id: Optional[str] = None

@dataclass
class CostEntry:
    """Cost tracking entry."""
    timestamp: str
    query_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

@dataclass
class PerformanceEntry:
    """Performance tracking entry."""
    timestamp: str
    query_id: str
    component: str
    operation: str
    duration_ms: float
    metadata: Dict[str, Any]

class ObservabilityManager:
    """Manages logging, cost tracking, and performance monitoring."""
    
    def __init__(self, log_file: str = None, enable_cost_tracking: bool = None):
        self.log_file = log_file or Config.LOG_FILE
        self.enable_cost_tracking = enable_cost_tracking if enable_cost_tracking is not None else Config.ENABLE_COST_TRACKING
        
        # Initialize logger
        self._setup_logger()
        
        # Cost tracking
        self.cost_entries: List[CostEntry] = []
        self.total_cost = 0.0
        
        # Performance tracking
        self.performance_entries: List[PerformanceEntry] = []
        
        # Query cache for cost tracking
        self.query_cache = {}
        
        # Create logs directory
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self):
        """Setup loguru logger with file and console output."""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=Config.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file handler
        logger.add(
            self.log_file,
            level=Config.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
    
    def log_info(self, message: str, component: str = "system", metadata: Dict[str, Any] = None, query_id: str = None):
        """Log info message."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level="INFO",
            component=component,
            message=message,
            metadata=metadata or {},
            query_id=query_id
        )
        logger.info(f"[{component}] {message}", extra=entry.__dict__)
    
    def log_warning(self, message: str, component: str = "system", metadata: Dict[str, Any] = None, query_id: str = None):
        """Log warning message."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level="WARNING",
            component=component,
            message=message,
            metadata=metadata or {},
            query_id=query_id
        )
        logger.warning(f"[{component}] {message}", extra=entry.__dict__)
    
    def log_error(self, message: str, component: str = "system", metadata: Dict[str, Any] = None, query_id: str = None):
        """Log error message."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level="ERROR",
            component=component,
            message=message,
            metadata=metadata or {},
            query_id=query_id
        )
        logger.error(f"[{component}] {message}", extra=entry.__dict__)
    
    def log_debug(self, message: str, component: str = "system", metadata: Dict[str, Any] = None, query_id: str = None):
        """Log debug message."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level="DEBUG",
            component=component,
            message=message,
            metadata=metadata or {},
            query_id=query_id
        )
        logger.debug(f"[{component}] {message}", extra=entry.__dict__)
    
    def track_cost(self, query_id: str, model_name: str, input_tokens: int, 
                   output_tokens: int, input_cost: float = None, output_cost: float = None):
        """Track cost for a query."""
        if not self.enable_cost_tracking:
            return
        
        # Get costs from config if not provided
        if input_cost is None or output_cost is None:
            model_costs = Config.get_model_costs().get(model_name, {"input": 0.0, "output": 0.0})
            input_cost = input_cost or (input_tokens * model_costs["input"] / 1000)
            output_cost = output_cost or (output_tokens * model_costs["output"] / 1000)
        
        total_tokens = input_tokens + output_tokens
        total_cost = input_cost + output_cost
        
        entry = CostEntry(
            timestamp=datetime.utcnow().isoformat(),
            query_id=query_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        self.cost_entries.append(entry)
        self.total_cost += total_cost
        
        self.log_info(
            f"Cost tracked: ${total_cost:.4f} for {total_tokens} tokens",
            component="cost_tracker",
            metadata={
                "query_id": query_id,
                "model_name": model_name,
                "total_cost": total_cost,
                "total_tokens": total_tokens
            },
            query_id=query_id
        )
    
    def track_performance(self, query_id: str, component: str, operation: str, 
                         duration_ms: float, metadata: Dict[str, Any] = None):
        """Track performance metrics."""
        entry = PerformanceEntry(
            timestamp=datetime.utcnow().isoformat(),
            query_id=query_id,
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self.performance_entries.append(entry)
        
        self.log_debug(
            f"Performance: {operation} took {duration_ms:.2f}ms",
            component=component,
            metadata={
                "query_id": query_id,
                "operation": operation,
                "duration_ms": duration_ms,
                **metadata
            },
            query_id=query_id
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        if not self.cost_entries:
            return {"total_cost": 0.0, "total_queries": 0, "average_cost_per_query": 0.0}
        
        total_queries = len(self.cost_entries)
        average_cost = self.total_cost / total_queries if total_queries > 0 else 0.0
        
        # Cost by model
        model_costs = {}
        for entry in self.cost_entries:
            model = entry.model_name
            if model not in model_costs:
                model_costs[model] = {"total_cost": 0.0, "queries": 0}
            model_costs[model]["total_cost"] += entry.total_cost
            model_costs[model]["queries"] += 1
        
        return {
            "total_cost": self.total_cost,
            "total_queries": total_queries,
            "average_cost_per_query": average_cost,
            "model_breakdown": model_costs
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_entries:
            return {"total_operations": 0, "average_duration_ms": 0.0}
        
        total_operations = len(self.performance_entries)
        average_duration = sum(entry.duration_ms for entry in self.performance_entries) / total_operations
        
        # Performance by component
        component_performance = {}
        for entry in self.performance_entries:
            component = entry.component
            if component not in component_performance:
                component_performance[component] = {"total_duration": 0.0, "operations": 0}
            component_performance[component]["total_duration"] += entry.duration_ms
            component_performance[component]["operations"] += 1
        
        # Calculate averages
        for component in component_performance:
            perf = component_performance[component]
            perf["average_duration"] = perf["total_duration"] / perf["operations"]
        
        return {
            "total_operations": total_operations,
            "average_duration_ms": average_duration,
            "component_breakdown": component_performance
        }
    
    def export_logs(self, output_file: str = None) -> str:
        """Export logs to JSON file."""
        output_file = output_file or f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "cost_summary": self.get_cost_summary(),
            "performance_summary": self.get_performance_summary(),
            "cost_entries": [asdict(entry) for entry in self.cost_entries],
            "performance_entries": [asdict(entry) for entry in self.performance_entries]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.log_info(f"Logs exported to {output_file}")
        return output_file
    
    def clear_logs(self):
        """Clear all logs."""
        self.cost_entries.clear()
        self.performance_entries.clear()
        self.total_cost = 0.0
        self.log_info("All logs cleared")

class PromptCache:
    """Prompt caching system."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or Config.PROMPT_CACHE_SIZE
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt."""
        # Create a hash of the prompt and parameters
        key_data = {"prompt": prompt, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """Get cached response."""
        key = self._generate_key(prompt, **kwargs)
        
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, prompt: str, response: Any, **kwargs):
        """Cache response."""
        key = self._generate_key(prompt, **kwargs)
        
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = response
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
