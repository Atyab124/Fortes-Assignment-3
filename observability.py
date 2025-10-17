"""Observability and monitoring for the RAG system."""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: datetime
    query_text: str
    response_time: float
    retrieval_time: float
    generation_time: float
    num_sources_retrieved: int
    max_similarity_score: float
    avg_similarity_score: float
    answer_length: int
    token_count: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time: float
    total_documents: int
    total_chunks: int
    total_embeddings: int
    memory_usage_mb: float
    cpu_usage_percent: float

class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, db_path: str = "metrics.db"):
        """Initialize metrics collector."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    query_text TEXT,
                    response_time REAL,
                    retrieval_time REAL,
                    generation_time REAL,
                    num_sources_retrieved INTEGER,
                    max_similarity_score REAL,
                    avg_similarity_score REAL,
                    answer_length INTEGER,
                    token_count INTEGER,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    total_queries INTEGER,
                    successful_queries INTEGER,
                    failed_queries INTEGER,
                    avg_response_time REAL,
                    total_documents INTEGER,
                    total_chunks INTEGER,
                    total_embeddings INTEGER,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL
                )
            """)
            
            # Performance logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    operation TEXT NOT NULL,
                    duration REAL,
                    success BOOLEAN,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_metrics (
                    query_id, timestamp, query_text, response_time, retrieval_time,
                    generation_time, num_sources_retrieved, max_similarity_score,
                    avg_similarity_score, answer_length, token_count, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.query_id,
                metrics.timestamp.isoformat(),
                metrics.query_text,
                metrics.response_time,
                metrics.retrieval_time,
                metrics.generation_time,
                metrics.num_sources_retrieved,
                metrics.max_similarity_score,
                metrics.avg_similarity_score,
                metrics.answer_length,
                metrics.token_count,
                metrics.success,
                metrics.error_message
            ))
            conn.commit()
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics (
                    timestamp, total_queries, successful_queries, failed_queries,
                    avg_response_time, total_documents, total_chunks, total_embeddings,
                    memory_usage_mb, cpu_usage_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.total_queries,
                metrics.successful_queries,
                metrics.failed_queries,
                metrics.avg_response_time,
                metrics.total_documents,
                metrics.total_chunks,
                metrics.total_embeddings,
                metrics.memory_usage_mb,
                metrics.cpu_usage_percent
            ))
            conn.commit()
    
    def record_performance_log(self, operation: str, duration: float, 
                             success: bool, metadata: Optional[Dict] = None):
        """Record performance log."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_logs (timestamp, operation, duration, success, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                operation,
                duration,
                success,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
    
    def get_query_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get query statistics for the last N hours."""
        since_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    AVG(response_time) as avg_response_time,
                    AVG(num_sources_retrieved) as avg_sources,
                    AVG(max_similarity_score) as avg_max_similarity,
                    AVG(avg_similarity_score) as avg_avg_similarity
                FROM query_metrics 
                WHERE timestamp >= ?
            """, (since_time.isoformat(),))
            
            stats = cursor.fetchone()
            
            # Response time distribution
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    CASE 
                        WHEN response_time < 1 THEN '< 1s'
                        WHEN response_time < 5 THEN '1-5s'
                        WHEN response_time < 10 THEN '5-10s'
                        ELSE '> 10s'
                    END as time_range
                FROM query_metrics 
                WHERE timestamp >= ?
                GROUP BY time_range
            """, (since_time.isoformat(),))
            
            response_time_dist = dict(cursor.fetchall())
            
            return {
                'period_hours': hours,
                'total_queries': stats[0] or 0,
                'successful_queries': stats[1] or 0,
                'failed_queries': (stats[0] or 0) - (stats[1] or 0),
                'success_rate': (stats[1] or 0) / (stats[0] or 1),
                'avg_response_time': stats[2] or 0,
                'avg_sources_retrieved': stats[3] or 0,
                'avg_max_similarity': stats[4] or 0,
                'avg_avg_similarity': stats[5] or 0,
                'response_time_distribution': response_time_dist
            }

class PerformanceMonitor:
    """Monitors system performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.start_times = {}
    
    @contextmanager
    def time_operation(self, operation_name: str, metrics_collector: Optional[MetricsCollector] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        success = True
        error = None
        
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time
            if metrics_collector:
                metrics_collector.record_performance_log(
                    operation_name, duration, success, {'error': error} if error else None
                )
    
    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if operation_name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        return duration

class TokenCounter:
    """Counts tokens in text (approximate)."""
    
    def __init__(self):
        """Initialize token counter."""
        # Simple token counting - can be enhanced with tiktoken
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def estimate_cost(self, tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost based on token count."""
        # Approximate costs (as of 2024)
        costs_per_1k_tokens = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "claude-3": 0.015
        }
        
        cost_per_1k = costs_per_1k_tokens.get(model, 0.002)
        return (tokens / 1000) * cost_per_1k

class RAGMonitor:
    """Main monitoring class for RAG system."""
    
    def __init__(self, rag_system, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize RAG monitor."""
        self.rag_system = rag_system
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.token_counter = TokenCounter()
        
        # Cache for system stats
        self._last_system_stats = None
        self._last_system_stats_time = None
    
    def monitor_query(self, query: str, result: Dict[str, Any], 
                     query_id: Optional[str] = None) -> QueryMetrics:
        """Monitor a query and record metrics."""
        if query_id is None:
            query_id = f"query_{int(time.time() * 1000)}"
        
        # Extract metrics from result
        response_time = result.get('query_time', 0.0)
        sources = result.get('sources', [])
        answer = result.get('answer', '')
        
        # Calculate similarity scores
        similarities = result.get('similarity_scores', [])
        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Estimate retrieval and generation times (approximate)
        retrieval_time = response_time * 0.3  # Assume 30% for retrieval
        generation_time = response_time * 0.7  # Assume 70% for generation
        
        # Count tokens
        token_count = self.token_counter.count_tokens(query + answer)
        
        # Create metrics
        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now(),
            query_text=query[:500],  # Truncate for storage
            response_time=response_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_sources_retrieved=len(sources),
            max_similarity_score=max_similarity,
            avg_similarity_score=avg_similarity,
            answer_length=len(answer),
            token_count=token_count,
            success=result.get('retrieval_success', False),
            error_message=result.get('error')
        )
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_query(metrics)
        
        return metrics
    
    def get_system_metrics(self, force_refresh: bool = False) -> SystemMetrics:
        """Get current system metrics."""
        now = datetime.now()
        
        # Use cache if available and not expired
        if (not force_refresh and self._last_system_stats and 
            self._last_system_stats_time and 
            (now - self._last_system_stats_time).seconds < 60):
            return self._last_system_stats
        
        # Get stats from RAG system
        rag_stats = self.rag_system.get_document_stats()
        
        # Get query stats from last hour
        query_stats = self.metrics_collector.get_query_stats(hours=1)
        
        # Get system resource usage (simplified)
        try:
            import psutil
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
        except ImportError:
            memory_usage = 0.0
            cpu_usage = 0.0
        
        metrics = SystemMetrics(
            timestamp=now,
            total_queries=query_stats['total_queries'],
            successful_queries=query_stats['successful_queries'],
            failed_queries=query_stats['failed_queries'],
            avg_response_time=query_stats['avg_response_time'],
            total_documents=rag_stats['total_documents'],
            total_chunks=rag_stats['total_chunks'],
            total_embeddings=rag_stats['total_embeddings'],
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
        
        # Cache the result
        self._last_system_stats = metrics
        self._last_system_stats_time = now
        
        # Record system metrics
        self.metrics_collector.record_system_metrics(metrics)
        
        return metrics
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        query_stats = self.metrics_collector.get_query_stats(hours)
        system_metrics = self.get_system_metrics()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'query_metrics': query_stats,
            'system_metrics': asdict(system_metrics),
            'recommendations': self._generate_recommendations(query_stats, system_metrics)
        }
    
    def _generate_recommendations(self, query_stats: Dict[str, Any], 
                                system_metrics: SystemMetrics) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Response time recommendations
        if query_stats['avg_response_time'] > 10:
            recommendations.append("Consider optimizing retrieval or generation for faster response times")
        
        # Success rate recommendations
        if query_stats['success_rate'] < 0.8:
            recommendations.append("Investigate failed queries to improve success rate")
        
        # Similarity score recommendations
        if query_stats['avg_max_similarity'] < 0.5:
            recommendations.append("Consider improving embedding quality or retrieval parameters")
        
        # Resource usage recommendations
        if system_metrics.memory_usage_mb > 1000:
            recommendations.append("Monitor memory usage - consider optimizing vector storage")
        
        if system_metrics.cpu_usage_percent > 80:
            recommendations.append("High CPU usage detected - consider load balancing")
        
        return recommendations
