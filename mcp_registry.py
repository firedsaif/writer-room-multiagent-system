import json
from typing import Dict, Any, Callable

class MCPRegistry:
    """
    Mock Model Context Protocol (MCP) Registry.
    Satisfies constraint: Agents query tool registry at runtime without hardcoded tool lists.
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}
        
    def register_tool(self, name: str, schema: dict, func: Callable):
        self._tools[name] = func
        self._schemas[name] = schema
        
    def discover_tools(self) -> list:
        """Returns structured JSON schemas for all registered tools."""
        return list(self._schemas.values())
        
    def execute_tool(self, name: str, inputs: dict) -> Any:
        """Executes a tool by name dynamically."""
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found in MCP Registry.")
        return self._tools[name](**inputs)

# Global singleton registry
mcp_registry = MCPRegistry()
