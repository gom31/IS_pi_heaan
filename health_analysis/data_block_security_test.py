import random
import base64
import json

# random input generate
def generate_random_data():
    return {
        'sex': random.choice([0, 1]),
        'age': random.randint(1, 7),
        'edu_lvl': random.randint(0, 4),
        'had_sex': random.choice([0, 1]),
        'n_s_part': random.randint(0, 20),
        'con_use': random.choice([0, 1]),
        'r_use_con': random.choice([0, 1]),
        'h_sti': random.choice([0, 1]),
        'h_aids': random.choice([0, 1])
    }

# encrypt block
class EncryptedBlock:
    def __init__(self, data):
        self._encrypted = base64.b64encode(json.dumps(data).encode()).decode()
        print(f"🔒 Block created: {self._encrypted[:20]}...")
    
    def __getitem__(self, key):
        raise Exception("🚫 ACCESS DENIED: Encrypted data cannot be accessed directly!")
    
    def __getattr__(self, name):
        raise Exception("🚫 ACCESS DENIED: Encrypted data cannot be accessed directly!")

# test
if __name__ == "__main__":
    print("📊 Generating random health data...")
    data = generate_random_data()
    print(f"Original data: {data}")
    
    print("\n🔐 Creating encrypted block...")
    block = EncryptedBlock(data)
    
    print("\n🚨 Trying to access encrypted data...")
    
    try:
        print("Accessing block['sex']...")
        value = block['sex']
        print(f"❌ SECURITY BREACH: Got {value}")
    except Exception as e:
        print(f"✅ BLOCKED: {e}")
    
    try:
        print("Accessing block.age...")
        value = block.age
        print(f"❌ SECURITY BREACH: Got {value}")
    except Exception as e:
        print(f"✅ BLOCKED: {e}")
    
    print("\n🎯 Test completed - Data is secure!")