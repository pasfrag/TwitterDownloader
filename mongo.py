from pymongo import MongoClient


class MongoHandler(object):
    client = None
    db = None

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["thesis_db"]

    # Stores a single instance to collection
    def store_to_collection(self, dictionary, col_name):
        collection = self.db[col_name]
        collection.insert_one(dictionary)

    # Retrieves a collection
    def retrieve_from_collection(self, col_name):
        collection = self.db[col_name]
        return collection.find(no_cursor_timeout=True)

    # Retrieves a single instance from a collection
    def get_with_id(self, col_name, dict_wit_id):
        collection = self.db[col_name]
        return collection.find(dict_wit_id)

    # Deletes a single instance from a collection
    def delete_with_name(self, col_name, name):
        collection = self.db[col_name]
        query = {"user_name": name}
        x = collection.delete_many(query)
        print(x.deleted_count, " documents deleted.")
