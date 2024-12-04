import json

def get_last_json_list(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        # Ensure the data is a list and return the last list
        if isinstance(data, list):
            return data[-1]
        else:
            raise ValueError("The JSON content is not a list.")
    except FileNotFoundError:
        print("Error: File not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None


# Example usage
file_path = "app/issue/selections.json"
last_list = get_last_json_list(file_path)
print(last_list)

print("########")
print(last_list.keys())
print(last_list['vehicleType'])
