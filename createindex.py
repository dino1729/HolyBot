import json
import time
import pinecone
import openai
import argparse
import logging
import dotenv
import os
import sys
import itertools

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_API_KEY"]
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_API_BASE")
openai.api_key = os.environ.get("AZURE_API_KEY")
azure_api_key = os.environ.get("AZURE_API_KEY")
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Create a logger
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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


class CreatePineconeIndex:

    def __init__(self):
        try:
            logger.info("Parsing arguments")
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("--holybook", type=str, required=True)
            args = parser.parse_args()

            self.holybook = args.holybook
            self.pinecone_apikey = pinecone_api_key
            self.pinecone_environment = pinecone_environment
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

    # initiates pinecone

    def init_pinecone(self):
        try:
            pinecone.init(api_key=self.pinecone_apikey,
                          environment=self.pinecone_environment)
        except Exception as e:
            logger.error("Error while initiating pinecone: {}".format(e))

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

    def insert_embeddings_pinecone(self, embeddings, data):
        try:
            self.init_pinecone()

            indexes = pinecone.list_indexes()

            if self.holybook in indexes:
                pinecone.delete_index(self.holybook)

            pinecone.create_index(self.holybook, dimension=len(embeddings[0]))

            # Connect to index
            index = pinecone.Index(self.holybook)

            keys = list(data.keys())

            to_upsert = [(str(keys[i]), list(embeddings[i]))
                         for i in range(len(embeddings))]

            # Upsert data with 100 vectors per upsert request
            for ids_vectors_chunk in self.chunks(to_upsert, batch_size=100):
                # Assuming `index` defined elsewhere
                index.upsert(vectors=ids_vectors_chunk)
            logger.info(
                f"Index on Pinecone for {self.holybook} created successfully")

        except Exception as e:
            logger.error("Error while inserting embeddings: {}".format(e))

    def createindex(self):
        try:
            logger.info(f"Reading holybook {self.holybook} data")
            # read data
            data = self.read_json()

            logger.info(
                f"Creating embeddings for holybook {self.holybook}")
            # create embeddings
            embeddings = self.create_embeddings(data)

            logger.info(
                f"Inserting embeddings into Pinecone for holybook {self.holybook}")
            # insert embeddings into pinecone
            self.insert_embeddings_pinecone(embeddings, data)
        except Exception as e:
            logger.error("Error while creating index: {}".format(e))


if __name__ == "__main__":
    PineConeIndex = CreatePineconeIndex()
    PineConeIndex.createindex()
