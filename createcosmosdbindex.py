import json
import time
import openai
import argparse
import logging
import dotenv
import os
from azure.cosmos import CosmosClient, PartitionKey
import itertools

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_API_KEY"]
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")
azure_api_key = os.environ.get("AZURE_API_KEY")
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
azure_cosmosdb_key = os.environ.get("AZURE_COSMOSDB_KEY")
azure_cosmosdb_uri = os.environ.get("AZURE_COSMOSDB_URI")

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add the formatter to the console handler
console_handler.setFormatter(formatter)
# Add the console handler to the logger
logger.addHandler(console_handler)


class CreateCosmosDBIndex:

    def __init__(self):
        try:
            logger.info("Parsing arguments")
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("--holybook", type=str, required=True)
            args = parser.parse_args()
            self.holybook = args.holybook
            self.azure_cosmosdb_key = azure_cosmosdb_key
            self.azure_cosmosdb_uri = azure_cosmosdb_uri
            self.openaikey = azure_api_key
        except Exception as e:
            logger.error("Error while parsing arguments: {}".format(e))

    def read_json(self):
        try:
            with open(f"{self.holybook}.json", "r") as f:
                data = json.loads(f.read())
            return data
        except Exception as e:
            logger.error("Error while reading json: {}".format(e))

    # creates openai embeddings

    @staticmethod
    def chunks(iterable, batch_size=100):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def create_embeddings(self, data):
        try:
            openai.api_key = self.openaikey
            embeddings = []
            for chunk in self.chunks(data, batch_size=1):
                response = openai.Embedding.create(
                    input=chunk,
                    engine="text-embedding-ada-002"
                )
                embeddings.extend([item['embedding'] for item in response['data']])
                time.sleep(1)
            return embeddings
        except Exception as e:
            logger.error("Error while creating embeddings: {}".format(e))

    # insert embeddings into pinecone
    def insert_embeddings_cosmosdb(self, embeddings, data):
        try:
            client = CosmosClient(self.azure_cosmosdb_uri,self.azure_cosmosdb_key)
            database_name = 'holybooks'
            container_name = self.holybook
            database = client.create_database_if_not_exists(id=database_name)
            container = database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/id"),
                offer_throughput=400
            )
            keys = list(data.keys())
            to_upsert = [{"id": str(keys[i]), "embedding": list(embeddings[i]), "data": data[keys[i]]}
                     for i in range(len(embeddings))]
            # Upsert data with 100 items per request
            for items_chunk in self.chunks(to_upsert, batch_size=100):
                # Use bulk operations to optimize performance
                for item in items_chunk:
                    container.upsert_item(body=item)
            logger.info(f"Index on CosmosDB for {self.holybook} created successfully")
        except Exception as e:
            logger.error("Error while inserting embeddings: {}".format(e))

    def createindex(self):
        try:
            logger.info(f"Reading holybook {self.holybook} data")
            # read data
            data = self.read_json()
            logger.info(f"Creating embeddings for holybook {self.holybook}")
            # create embeddings
            embeddings = self.create_embeddings(data)
            logger.info(f"Inserting embeddings into CosmosDB for holybook {self.holybook}")
            # insert embeddings into cosmosdb
            self.insert_embeddings_cosmosdb(embeddings, data)
        except Exception as e:
            logger.error("Error while creating index: {}".format(e))

if __name__ == "__main__":
    CosmosDBIndex = CreateCosmosDBIndex()
    CosmosDBIndex.createindex()
