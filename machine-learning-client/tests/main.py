#Runs a continuous polling loop connected to MongoDB via pymongo. 
# When it finds a session marked "ready_for_ml", it runs detector.py on the saved frames 
# and writes the blink_count back to MongoDB.