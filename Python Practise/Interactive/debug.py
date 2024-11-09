def calculate_discount(price, discount_percent):
    # Applies a discount and returns the final price

    discount = price * (discount_percent / 100)
    final_price = price - discount
    return round(final_price, 2)

def checkout(prices, discounts):
    total = 0
    logs=[]
    logs.append(f"length of prices arrauy is {len(prices)}")
    logs.append(f"yes")
    print("Length of prices array ",len(prices))
    print("length of discount arrauy",len(discounts))
    for i in range(len(prices)):
        total += calculate_discount(prices[i], discounts[i])
    
    with open("debug.txt","a") as log_file:
        for log in logs:
            log_file.write(log+"\n")
            
    return total

# Example usage:
prices = [100, 200, 300]
discounts = [10, 20,10]  # Missing a discount value for the last item, should fail
print("Total after discounts:", checkout(prices, discounts))