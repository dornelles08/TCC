from pymongo import MongoClient

client = MongoClient("mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC?retryWrites=true&w=majority")
db = client.anuncios

# anuncios = db.find()

# print(anuncios)