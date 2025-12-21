from typing import List, Dict


# Build graph in memory
_BUS_ROUTES_GRAPH = None


def _build_graph() -> Dict[str, List[str]]:
    """Build Karachi bus routes graph in memory as a bidirectional (cyclic) graph."""
    routes = [
        ["saddar", "tariq road", "shahrah-e-faisal", "gulshan-e-iqbal", "nipa"],
        ["nazimabad", "liaquatabad", "shahrah-e-faisal", "clifton", "do darya"],
        ["orangi", "nazimabad", "liaquatabad", "saddar", "ii chundrigar road"],
        ["gulshan-e-iqbal", "shahrah-e-faisal", "clifton", "seaview"],
        ["korangi", "shahrah-e-faisal", "tariq road", "saddar"],
        ["malir", "airport", "shahrah-e-faisal", "saddar"],
        ["landhi", "korangi", "shahrah-e-faisal", "gulshan-e-iqbal"],
        ["saddar", "liaquatabad", "nazimabad", "orangi", "saddar"],
        ["clifton", "shahrah-e-faisal", "gulshan-e-iqbal", "nipa", "gulshan-e-iqbal"],
        ["airport", "shahrah-e-faisal", "tariq road", "saddar", "ii chundrigar road"],
    ]
    
    graph = {}
    for route in routes:
        route = [stop.lower().strip() for stop in route]
        for i in range(len(route) - 1):
            current = route[i]
            next_stop = route[i + 1]
            
            # Initialize nodes if they don't exist
            if current not in graph:
                graph[current] = []
            if next_stop not in graph:
                graph[next_stop] = []
            
            # Add forward edge (current -> next_stop)
            if next_stop not in graph[current]:
                graph[current].append(next_stop)
            
            # Add reverse edge (next_stop -> current) to make it bidirectional
            if current not in graph[next_stop]:
                graph[next_stop].append(current)
    
    return graph


def find_route(source: str, destination: str, depth: int) -> List[List[str]]:
    """
    Find all routes from source to destination with depth limit.
    
    Args:
        source: Starting bus stop (converted to lowercase)
        destination: Target bus stop (converted to lowercase)
        depth: Maximum number of transfers/intermediate stops allowed (prevents infinite loops)
               depth=0 means direct connection, depth=1 means 1 transfer, etc.
        
    Returns:
        List of routes, where each route is a list of stops from source to destination
    """
    global _BUS_ROUTES_GRAPH
    if _BUS_ROUTES_GRAPH is None:
        _BUS_ROUTES_GRAPH = _build_graph()
    
    source = source.lower().strip()
    destination = destination.lower().strip()
    graph = _BUS_ROUTES_GRAPH
    
    # Check if both nodes exist in the graph
    if source not in graph or destination not in graph:
        return []
    
    if source == destination:
        return [[source]]
    
    all_routes = []
    
    def search(current: str, target: str, max_transfers: int, path: List[str], visited: set):
        path.append(current)
        visited.add(current)
        
        if current == target:
            all_routes.append(path.copy())
        elif max_transfers > 0:
            # max_transfers represents remaining transfers allowed
            # We can explore neighbors if we have transfers remaining
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        search(neighbor, target, max_transfers - 1, path, visited)
        elif max_transfers == 0:
            # Last transfer: check if destination is directly reachable
            if current in graph and target in graph[current] and target not in visited:
                path.append(target)
                all_routes.append(path.copy())
                path.pop()
        
        path.pop()
        visited.remove(current)
    
    search(source, destination, depth, [], set())
    return all_routes

_BUS_ROUTES_GRAPH = _build_graph()
# function to get all route name as a unique list
def get_all_route_names() -> List[str]:
    return list(set([stop for route in _BUS_ROUTES_GRAPH.values() for stop in route]))

# print(get_all_route_names())
print(find_route("nipa", "do darya", 5))