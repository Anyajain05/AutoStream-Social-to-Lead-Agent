print("Importing server.app...")
try:
    from server import app
    print("Server imported successfully!")
except Exception as e:
    import traceback
    print("Error importing server:")
    traceback.print_exc()
