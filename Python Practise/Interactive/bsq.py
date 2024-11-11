class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, price, quantity):
        # Add or update an item in the inventory
        if name in self.items:
            self.items[name]['quantity'] += quantity
        else:
            self.items[name] = {'price': price, 'quantity': quantity}

    def remove_item(self, name, quantity):
        # Remove items from the inventory
        if name in self.items:
            if self.items[name]['quantity'] >= quantity:
                self.items[name]['quantity'] -= quantity
            else:
                print(f"Error: Not enough {name} in inventory")
        else:
            print(f"Error: Item {name} does not exist in inventory")

    def total_inventory_value(self):
        # Calculate the total value of the inventory
        total_value = 0
        for item, details in self.items.items():
            total_value += details['price'] * details['quantity']
        return total_value

    def check_item(self, name):
        # Check the available quantity of an item
        if name in self.items:
            return self.items[name]['quantity']
        else:
            print(f"Error: Item {name} does not exist in inventory")
            return None

# Example usage
inventory = Inventory()

# Adding items
inventory.add_item("Apple", 0.5, 100)
inventory.add_item("Banana", 0.3, 150)
inventory.add_item("Orange", 0.8, 80)

# Removing items
inventory.remove_item("Apple", 50)
inventory.remove_item("Banana", 200)  # Error case, quantity exceeds

# Checking total inventory value
print("Total inventory value:", inventory.total_inventory_value())

# Checking specific item
print("Orange quantity:", inventory.check_item("Orange"))