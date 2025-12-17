"""
Virtuoso Query Compiler and Execution Engine
=============================================
Custom query compilation unit that translates and optimizes queries for Virtuoso.

Features:
- SQL to Virtuoso-optimized SQL compilation
- Query plan generation and caching
- Prepared statement compilation
- Batch query optimization
- Query result caching
- Execution plan visualization
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta


class CompilationMode(Enum):
    """Query compilation modes"""
    STANDARD = "standard"           # Standard SQL compilation
    OPTIMIZED = "optimized"         # Apply all optimizations
    CACHED = "cached"               # Use cached execution plans
    BATCH = "batch"                 # Batch multiple queries
    PARALLEL = "parallel"           # Enable parallel execution hints


class QueryType(Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"
    STORED_PROC = "CALL"


@dataclass
class CompiledQuery:
    """Represents a compiled query ready for execution"""
    query_id: str
    original_sql: str
    compiled_sql: str
    query_type: QueryType
    parameters: List[str] = field(default_factory=list)
    execution_plan: Optional[str] = None
    estimated_rows: int = 0
    estimated_cost: float = 0.0
    cache_key: Optional[str] = None
    compile_time: datetime = field(default_factory=datetime.now)
    optimizations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query_id': self.query_id,
            'original_sql': self.original_sql,
            'compiled_sql': self.compiled_sql,
            'query_type': self.query_type.value,
            'parameters': self.parameters,
            'execution_plan': self.execution_plan,
            'estimated_rows': self.estimated_rows,
            'estimated_cost': self.estimated_cost,
            'optimizations': self.optimizations,
            'compile_time': self.compile_time.isoformat()
        }


@dataclass
class BatchQuery:
    """Batch of queries to execute together"""
    queries: List[CompiledQuery]
    batch_id: str
    total_estimated_cost: float = 0.0
    execution_order: List[int] = field(default_factory=list)


class QueryCompiler:
    """
    Advanced query compiler for Virtuoso
    Compiles, optimizes, and prepares queries for execution
    """
    
    def __init__(self, connection, enable_caching: bool = True):
        """
        Initialize query compiler
        
        Args:
            connection: Database connection
            enable_caching: Enable query plan caching
        """
        self.conn = connection
        self.enable_caching = enable_caching
        self.compiled_cache: Dict[str, CompiledQuery] = {}
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def compile(self, sql: str, mode: CompilationMode = CompilationMode.OPTIMIZED,
                parameters: Optional[List[Any]] = None) -> CompiledQuery:
        """
        Compile a SQL query
        
        Args:
            sql: SQL query string
            mode: Compilation mode
            parameters: Query parameters for prepared statements
            
        Returns:
            CompiledQuery object
        """
        # Generate cache key
        cache_key = self._generate_cache_key(sql, parameters)
        
        # Check cache
        if self.enable_caching and cache_key in self.compiled_cache:
            self.cache_hits += 1
            return self.compiled_cache[cache_key]
        
        self.cache_misses += 1
        
        # Detect query type
        query_type = self._detect_query_type(sql)
        
        # Compile based on mode
        if mode == CompilationMode.STANDARD:
            compiled_sql = self._standard_compile(sql)
        elif mode == CompilationMode.OPTIMIZED:
            compiled_sql = self._optimized_compile(sql)
        elif mode == CompilationMode.CACHED:
            compiled_sql = self._cached_compile(sql)
        elif mode == CompilationMode.PARALLEL:
            compiled_sql = self._parallel_compile(sql)
        else:
            compiled_sql = sql
        
        # Create compiled query object
        query_id = self._generate_query_id(sql)
        compiled = CompiledQuery(
            query_id=query_id,
            original_sql=sql,
            compiled_sql=compiled_sql,
            query_type=query_type,
            parameters=parameters or [],
            cache_key=cache_key
        )
        
        # Analyze and add execution plan
        if query_type == QueryType.SELECT:
            compiled.execution_plan = self._generate_execution_plan(compiled_sql)
            compiled.estimated_rows, compiled.estimated_cost = self._estimate_query_cost(compiled_sql)
        
        # Cache the compiled query
        if self.enable_caching:
            self.compiled_cache[cache_key] = compiled
        
        return compiled
    
    def _generate_cache_key(self, sql: str, parameters: Optional[List[Any]] = None) -> str:
        """Generate unique cache key for query"""
        key_data = sql
        if parameters:
            key_data += str(parameters)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_query_id(self, sql: str) -> str:
        """Generate unique query ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{sql}{timestamp}".encode()).hexdigest()[:16]
    
    def _detect_query_type(self, sql: str) -> QueryType:
        """Detect the type of SQL query"""
        sql_upper = sql.strip().upper()
        
        for query_type in QueryType:
            if sql_upper.startswith(query_type.value):
                return query_type
        
        return QueryType.SELECT  # Default
    
    def _standard_compile(self, sql: str) -> str:
        """Standard compilation - minimal changes"""
        # Just normalize whitespace
        return ' '.join(sql.split())
    
    def _optimized_compile(self, sql: str) -> str:
        """Optimized compilation with all optimizations"""
        compiled = sql
        
        # Normalize
        compiled = self._normalize_sql(compiled)
        
        # Apply optimizations
        compiled = self._apply_predicate_optimization(compiled)
        compiled = self._apply_join_optimization(compiled)
        compiled = self._apply_index_hints(compiled)
        compiled = self._apply_virtuoso_hints(compiled)
        
        return compiled
    
    def _cached_compile(self, sql: str) -> str:
        """Compile with result caching hints"""
        compiled = self._optimized_compile(sql)
        
        # Add Virtuoso result cache hint
        if 'OPTION' not in compiled.upper():
            compiled += ' OPTION (RESULT_CACHE)'
        
        return compiled
    
    def _parallel_compile(self, sql: str) -> str:
        """Compile with parallel execution hints"""
        compiled = self._optimized_compile(sql)
        
        # Add parallel execution hints
        if 'OPTION' not in compiled.upper():
            compiled += ' OPTION (PARALLEL)'
        else:
            compiled = compiled.replace('OPTION (', 'OPTION (PARALLEL, ')
        
        return compiled
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL formatting"""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.strip()
        
        # Uppercase SQL keywords
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE', 'BETWEEN',
            'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'OFFSET',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
            'AS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'
        ]
        
        for keyword in keywords:
            sql = re.sub(rf'\b{keyword}\b', keyword, sql, flags=re.IGNORECASE)
        
        return sql
    
    def _apply_predicate_optimization(self, sql: str) -> str:
        """Optimize WHERE clause predicates"""
        # Move most selective predicates first
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|OPTION|$)', 
                               sql, re.IGNORECASE)
        if not where_match:
            return sql
        
        where_clause = where_match.group(1).strip()
        
        # Split on AND
        predicates = [p.strip() for p in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE)]
        
        # Sort by estimated selectivity
        def selectivity(pred: str) -> int:
            if '=' in pred and '.id' in pred.lower():
                return 0  # Primary key equality
            elif '=' in pred:
                return 1  # Equality
            elif any(op in pred for op in ['<', '>', 'BETWEEN']):
                return 2  # Range
            elif 'IN' in pred.upper():
                return 3  # IN clause
            else:
                return 4  # Other
        
        optimized = sorted(predicates, key=selectivity)
        new_where = ' AND '.join(optimized)
        
        sql = sql.replace(where_clause, new_where)
        return sql
    
    def _apply_join_optimization(self, sql: str) -> str:
        """Optimize JOIN operations"""
        # Ensure smaller tables are joined first (if we have stats)
        # For now, just ensure INNER JOINs are explicit
        sql = re.sub(r'\bJOIN\b', 'INNER JOIN', sql, flags=re.IGNORECASE)
        return sql
    
    def _apply_index_hints(self, sql: str) -> str:
        """Add index usage hints"""
        # Virtuoso uses INDEX hints differently
        # We'll add TABLE OPTION hints for indexed columns
        
        # Example: games USE INDEX (idx_games_name)
        # For now, just ensure indexed columns are used in WHERE
        return sql
    
    def _apply_virtuoso_hints(self, sql: str) -> str:
        """Add Virtuoso-specific optimization hints"""
        hints = []
        
        # Check if ORDER BY is present
        if 'ORDER BY' in sql.upper():
            hints.append('ORDER')
        
        # Check if complex aggregations
        if any(func in sql.upper() for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
            if 'GROUP BY' in sql.upper():
                hints.append('HASH_GROUP')
        
        # Add hints if any
        if hints and 'OPTION' not in sql.upper():
            sql += f" OPTION ({', '.join(hints)})"
        
        return sql
    
    def _generate_execution_plan(self, sql: str) -> Optional[str]:
        """Generate execution plan for query"""
        try:
            cursor = self.conn.cursor()
            # Virtuoso EXPLAIN
            cursor.execute(f"EXPLAIN {sql}")
            plan = []
            for row in cursor.fetchall():
                plan.append(str(row))
            cursor.close()
            return '\n'.join(plan) if plan else None
        except Exception as e:
            return f"Error generating plan: {e}"
    
    def _estimate_query_cost(self, sql: str) -> Tuple[int, float]:
        """
        Estimate query cost and row count
        
        Returns:
            Tuple of (estimated_rows, estimated_cost)
        """
        # Simple heuristic-based estimation
        cost = 10.0
        rows = 100
        
        # Count tables
        tables = len(re.findall(r'\bFROM\b|\bJOIN\b', sql, re.IGNORECASE))
        cost += tables * 50
        rows *= tables * 10
        
        # Aggregations
        if 'GROUP BY' in sql.upper():
            cost += 30
            rows //= 2
        
        # Sorting
        if 'ORDER BY' in sql.upper():
            cost += 20
        
        # Subqueries
        subqueries = sql.upper().count('SELECT') - 1
        cost += subqueries * 100
        
        return rows, cost
    
    def compile_batch(self, queries: List[str], 
                     mode: CompilationMode = CompilationMode.OPTIMIZED) -> BatchQuery:
        """
        Compile multiple queries as a batch
        
        Args:
            queries: List of SQL queries
            mode: Compilation mode
            
        Returns:
            BatchQuery object
        """
        compiled_queries = []
        total_cost = 0.0
        
        for sql in queries:
            compiled = self.compile(sql, mode)
            compiled_queries.append(compiled)
            total_cost += compiled.estimated_cost
        
        # Optimize execution order (lowest cost first for better resource usage)
        execution_order = sorted(range(len(compiled_queries)), 
                               key=lambda i: compiled_queries[i].estimated_cost)
        
        batch_id = hashlib.md5(str(queries).encode()).hexdigest()[:16]
        
        return BatchQuery(
            queries=compiled_queries,
            batch_id=batch_id,
            total_estimated_cost=total_cost,
            execution_order=execution_order
        )
    
    def execute_compiled(self, compiled: CompiledQuery, 
                        fetch_results: bool = True) -> Optional[List[Any]]:
        """
        Execute a compiled query
        
        Args:
            compiled: CompiledQuery object
            fetch_results: Whether to fetch and return results
            
        Returns:
            Query results or None
        """
        cursor = self.conn.cursor()
        
        try:
            start_time = datetime.now()
            
            # Execute compiled SQL
            if compiled.parameters:
                cursor.execute(compiled.compiled_sql, compiled.parameters)
            else:
                cursor.execute(compiled.compiled_sql)
            
            # Fetch results if SELECT
            results = None
            if fetch_results and compiled.query_type == QueryType.SELECT:
                results = cursor.fetchall()
            
            # Record execution stats
            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_execution_stats(compiled.query_id, execution_time, 
                                        len(results) if results else 0)
            
            return results
            
        except Exception as e:
            self._record_execution_stats(compiled.query_id, 0, 0, error=str(e))
            raise
        finally:
            cursor.close()
    
    def execute_batch(self, batch: BatchQuery) -> List[Optional[List[Any]]]:
        """
        Execute a batch of queries
        
        Args:
            batch: BatchQuery object
            
        Returns:
            List of results for each query
        """
        results = []
        
        for idx in batch.execution_order:
            query = batch.queries[idx]
            result = self.execute_compiled(query)
            results.append(result)
        
        return results
    
    def _record_execution_stats(self, query_id: str, execution_time: float,
                               rows_returned: int, error: Optional[str] = None):
        """Record query execution statistics"""
        self.execution_stats[query_id] = {
            'execution_time': execution_time,
            'rows_returned': rows_returned,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.compiled_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear the compiled query cache"""
        self.compiled_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def export_compiled_query(self, compiled: CompiledQuery, filepath: str):
        """Export compiled query to file"""
        with open(filepath, 'w') as f:
            json.dump(compiled.to_dict(), f, indent=2)
    
    def get_execution_report(self) -> str:
        """Generate execution statistics report"""
        report = []
        report.append("=" * 80)
        report.append("QUERY COMPILER EXECUTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Cache stats
        cache_stats = self.get_cache_stats()
        report.append("CACHE STATISTICS:")
        report.append("-" * 80)
        report.append(f"  Cache Size: {cache_stats['cache_size']} queries")
        report.append(f"  Cache Hits: {cache_stats['cache_hits']}")
        report.append(f"  Cache Misses: {cache_stats['cache_misses']}")
        report.append(f"  Hit Rate: {cache_stats['hit_rate']}")
        report.append("")
        
        # Execution stats
        if self.execution_stats:
            report.append("RECENT EXECUTIONS:")
            report.append("-" * 80)
            
            # Sort by execution time
            sorted_stats = sorted(
                self.execution_stats.items(),
                key=lambda x: x[1]['execution_time'],
                reverse=True
            )
            
            for query_id, stats in sorted_stats[:10]:  # Top 10
                report.append(f"  Query ID: {query_id}")
                report.append(f"    Time: {stats['execution_time']:.4f}s")
                report.append(f"    Rows: {stats['rows_returned']}")
                if stats['error']:
                    report.append(f"    Error: {stats['error']}")
                report.append("")
        
        report.append("=" * 80)
        return '\n'.join(report)
    
    def visualize_execution_plan(self, compiled: CompiledQuery) -> str:
        """
        Create a visual representation of the execution plan
        
        Args:
            compiled: CompiledQuery with execution plan
            
        Returns:
            Formatted execution plan
        """
        if not compiled.execution_plan:
            return "No execution plan available"
        
        visual = []
        visual.append("=" * 80)
        visual.append(f"EXECUTION PLAN for Query {compiled.query_id}")
        visual.append("=" * 80)
        visual.append("")
        visual.append("Original Query:")
        visual.append(compiled.original_sql)
        visual.append("")
        visual.append("Compiled Query:")
        visual.append(compiled.compiled_sql)
        visual.append("")
        visual.append("Execution Plan:")
        visual.append("-" * 80)
        visual.append(compiled.execution_plan)
        visual.append("")
        visual.append(f"Estimated Rows: {compiled.estimated_rows}")
        visual.append(f"Estimated Cost: {compiled.estimated_cost:.2f}")
        visual.append("")
        
        if compiled.optimizations:
            visual.append("Applied Optimizations:")
            for opt in compiled.optimizations:
                visual.append(f"  • {opt}")
            visual.append("")
        
        visual.append("=" * 80)
        return '\n'.join(visual)


# Example usage
if __name__ == "__main__":
    print("Virtuoso Query Compiler Module")
    print("=" * 80)
    print("\nFeatures:")
    print("  • Query compilation and optimization")
    print("  • Execution plan generation")
    print("  • Query caching for performance")
    print("  • Batch query processing")
    print("  • Execution statistics tracking")
    print("\nUsage example:")
    print("""
    from virtuoso_query_compiler import QueryCompiler, CompilationMode
    import pyodbc
    
    # Connect to Virtuoso
    conn = pyodbc.connect(
        'DRIVER={virtuoso-odbc};HOST=localhost;PORT=1111;UID=dba;PWD=dba'
    )
    
    # Create compiler
    compiler = QueryCompiler(conn, enable_caching=True)
    
    # Compile and execute query
    sql = "SELECT * FROM games WHERE average_fps < 60 ORDER BY name"
    compiled = compiler.compile(sql, mode=CompilationMode.OPTIMIZED)
    results = compiler.execute_compiled(compiled)
    
    # View execution plan
    print(compiler.visualize_execution_plan(compiled))
    
    # Get statistics
    print(compiler.get_execution_report())
    """)
