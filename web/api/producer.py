import pika
import json


credentials = pika.PlainCredentials('admin', 'mypass')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbit', credentials=credentials))
channel = connection.channel()

channel.queue_declare(queue='hello')

def publish(body):
    print("body: ", body)
    channel.basic_publish(exchange='', routing_key='hello', body=json.dumps(body))
    return True
