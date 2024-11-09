from collections import defaultdict

def parse_shipping_details(input_string):
    """Parse the input string to create a dictionary of direct conversions."""
    shipping_dict = defaultdict(list)
    shipping_details = input_string.split(",")
    
    for detail in shipping_details:
        origin, destination, via, cost = detail.split(":")
        cost = int(cost)
        shipping_dict[origin].append((destination, cost, via))
    
    return shipping_dict

def direct_conversion_cost(shipping_dict, origin, destination):
    """Get direct conversion cost from origin to destination, if available."""
    if origin in shipping_dict:
        for dest, cost, via in shipping_dict[origin]:
            if dest == destination:
                return cost, via
    return None, "Direct conversion not available"

def indirect_conversion_cost(shipping_dict, origin, destination):
    """Get minimum cost with at most one intermediate hop from origin to destination."""
    min_cost = float('inf')
    best_route = None
    
    if origin in shipping_dict:
        # Check if there is a direct route first
        direct_cost, direct_via = direct_conversion_cost(shipping_dict, origin, destination)
        if direct_cost is not None:
            min_cost, best_route = direct_cost, [direct_via]
        
        # Check indirect routes
        for intermediate_dest, cost1, via1 in shipping_dict[origin]:
            if intermediate_dest in shipping_dict:
                for final_dest, cost2, via2 in shipping_dict[intermediate_dest]:
                    if final_dest == destination:
                        total_cost = cost1 + cost2
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_route = [via1, via2]
    
    if best_route:
        return min_cost, best_route
    else:
        return None, "Indirect conversion not available"

# Main execution
if __name__ == '__main__':
    input_string = input("Enter the details (e.g., 'USD:CAD:DHL:5,USD:GBP:FEDX:10'): ")
    shipping_dict = parse_shipping_details(input_string)
    
    input_type = input("Enter 1 for direct shipping and 0 for indirect shipping: ")
    if input_type == '1':
        origin, destination = input("Enter origin and destination separated by a comma (e.g., 'USD,CAD'): ").split(",")
        cost, method = direct_conversion_cost(shipping_dict, origin, destination)
        if cost:
            print(f"Direct shipping cost: {cost}, Method: {method}")
        else:
            print(method)
    elif input_type == '0':
        origin, destination = input("Enter origin and destination separated by a comma (e.g., 'USD,GBP'): ").split(",")
        cost, methods = indirect_conversion_cost(shipping_dict, origin, destination)
        if cost:
            print(f"Indirect shipping minimum cost: {cost}, Methods involved: {methods}")
        else:
            print(methods)
    else:
        print("Invalid input type selected.")