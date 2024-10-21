class Solution:
    def solve(self, input_str) -> dict:
        travel_dict = {}
        input_strings = input_str.split(":")
        
        for input_str in input_strings:
            input_str = input_str.split(",")
            origin = input_str[0]
            destination = input_str[1]
            cost = int(input_str[3])
            
            if origin not in travel_dict:
                travel_dict[origin] = []
            
            travel_dict[origin].append((destination, cost))
        
        return travel_dict

    def find_direct_route(self, travel_dict, origin, destination) -> int:
        if origin not in travel_dict:
            return float('inf')  # No direct route found
        
        for dest, cost in travel_dict[origin]:
            if dest == destination:
                return cost
        
        return float('inf')  # No direct route found

    def find_indirect_route(self, travel_dict, origin, destination) -> int:
        if origin not in travel_dict:
            return float('inf')  # No route found
        
        # Check for two-step routes: origin -> intermediate -> destination
        for intermediate, cost1 in travel_dict[origin]:
            if intermediate in travel_dict:
                for dest, cost2 in travel_dict[intermediate]:
                    if dest == destination:
                        return cost1 + cost2
        
        return float('inf')  # No valid two-hop route found

if __name__ == '__main__':
    solutionClass = Solution()
    
    input_places = input("Enter the details: ")
    travel_dict = solutionClass.solve(input_places)
    
    enter_choice = input("Enter 1 for direct distance and 0 for indirect distance: ")
    input_place = input("Enter places separated by comma: ").split(",")
    
    if enter_choice == "1":
        result = solutionClass.find_direct_route(travel_dict, input_place[0], input_place[1])
    else:
        result = solutionClass.find_indirect_route(travel_dict, input_place[0], input_place[1])
    
    if result == float('inf'):
        print("Route not found")
    else:
        print(f"Cost of route: {result}")
