#!/usr/bin/env python
import pika, sys, os
import json
import time
from app import render_video_from_strips
from sequancer.render import render
import logging

def main():
    credentials = pika.PlainCredentials('admin', 'mypass')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbit', credentials=credentials))
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        data = json.loads(body.decode('utf8'))
        
        print(data)
        print(" [x] start processing")
        #list_strip = render(json_data=data)
        #output_path = render_video_from_strips(list_strip, video_id=data['video_id'])
        # print(" [x] processed: ", output_path)

        # send message to another queue
        # channel.basic_publish(exchange='', routing_key='hello2', body=output_path)

    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    logging.info(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == '__main__':
    try:
        logging.info("start")
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)