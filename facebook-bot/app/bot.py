import logging.config
from threading import Thread
from time import sleep
import settings
import utils
import random
import os
import shutil
import requests
import predict
from pymessenger.bot import Bot
from flask import Flask, request

# create Flask application
app = Flask(__name__)

bot = Bot(os.environ["ACCESS_TOKEN"])

def sendInstructions(recipient_id):
    response_sent_text = "Initializing..."
    send_message(recipient_id, response_sent_text)
    response_sent_text = "Starting instruction protocol..."
    send_message(recipient_id, response_sent_text)
    sleep(2)
    response_sent_text = "1. Please open Facebook in another tab and disable your adblocker.\n2. Refresh Facebook.\n3. Make a screenshot of your newsfeed. Please note: This only works on Facebook in your browser, not mobile."
    send_message(recipient_id, response_sent_text)
    response_sent_text = "Starting privacy protocol..."
    send_message(recipient_id, response_sent_text)
    sleep(3)
    response_sent_text = "When you send me a screenshot of your Facebook feed, it might contain pictures of your friends' posts or other things you would like to keep private. Please double check your screenshot before sending. When I receive it, I will try to detect ads and then delete it. For more information on your privacy rights, please refer to our privacy policy https://adblockplus.org/en/privacy."
    send_message(recipient_id, response_sent_text)
    response_sent_text = "Starting detection protocol..."
    sleep(5)
    send_message(recipient_id, response_sent_text)
    response_sent_text = "Please send your screenshot"
    send_message(recipient_id, response_sent_text)    

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
       separator = "#"
       output = request.get_json()
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('postback'):
                recipient_id = message['sender']['id']
                if message['postback']['payload'].split(separator)[0] == 'incorrect_guess':
                    response_text = 'May I keep the screenshot to train and become better? Your privacy is important - I will only use your screenshot to make my ad detection better. If yes, please make sure you are not sharing any sensitive data'
                    ret = bot.send_button_message(recipient_id, response_text,
                        [
                            {
                            "type": "postback",
                            "title": "No, delete my file",
                            "payload": "do_not_keep_screenshot" + separator + 
                            message['postback']['payload'].split(separator)[1]
                            },
                            {
                            "type": "postback",
                            "title": "Yes, use my file",
                            "payload": "keep_screenshot" + separator +
                            message['postback']['payload'].split(separator)[1]
                            }                                            
                        ])

                if message['postback']['payload'].split(separator)[0] == 'keep_screenshot':
                    response_sent_text = "Thank you!"
                    send_message(recipient_id, response_sent_text)
                    response_sent_text = "Signing off..."
                    send_message(recipient_id, response_sent_text)

                if message['postback']['payload'].split(separator)[0] == 'do_not_keep_screenshot':
                    os.remove(message['postback']['payload'].split(separator)[1])
                    response_sent_text = "Ok, deleted!"
                    send_message(recipient_id, response_sent_text)
                    response_sent_text = "Signing off..."
                    send_message(recipient_id, response_sent_text)
                    
                if message['postback']['payload'].split(separator)[0] == 'correct_guess':
                    os.remove(message['postback']['payload'].split(separator)[1])
                    response_sent_text = "Great, thank you!"
                    send_message(recipient_id, response_sent_text)
                    response_sent_text = "Signing off..."
                    send_message(recipient_id, response_sent_text)
                    
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']
                usr_path = os.path.join('/submitted', recipient_id)
                
                if message['message'].get('text'):
                    if message['message'].get('text').lower() == "delete my data":
                        if os.path.exists(usr_path):
                            shutil.rmtree(usr_path)
                            response_sent_text = "Ok, deleted your data. Thanks for playing!"
                        else:
                            response_sent_text = "I don't have any data stored for you"
                        send_message(recipient_id, response_sent_text)
                        return "Message processed"
                        
                    thread = Thread(target = sendInstructions, args = (recipient_id,))
                    thread.start()                    

                #if user sends us a GIF, photo,video, or any other non-text item
                if message['message'].get('attachments'):
                    if message['message'].get('attachments')[0]['type'] != 'image':
                        send_message(recipient_id, "This doesn't look like an image. Please send me a screenshot of a dektop Facebook")
                        return "Message Processed"

                    ret = bot.send_action(recipient_id, "mark_seen")
                    ret = bot.send_action(recipient_id, "typing")

                    image_url = message['message'].get('attachments')[0]['payload']['url']
                    img_data = requests.get(image_url).content

                    try:
                        img_data, boxes = predict.predict_and_draw_boxes(img_data)
                    except Exception as e:
                        send_message(recipient_id, "Ooops! I ran into a prediction error. Try another image maybe?")
                        return "Message processed"

                    if (not os.path.exists(usr_path)):
                        os.makedirs(usr_path)

                    image_path = os.path.join(usr_path, "screenshot.jpg")
                    num = 1
                    while os.path.exists(image_path):
                        image_path = os.path.join(usr_path, "screenshot" + str(num) + ".jpg")
                        num = num + 1

                    with open(image_path, 'wb') as handler:
                        handler.write(img_data)

                    if len(boxes) == 0:
                        response_text = "I did not identify any Facebook newsfeed ads there. Is this correct?"
                    else:
                        ret = bot.send_image(recipient_id, image_path)
                        response_text = "Did I identify the ads? Go here to zoom in: https://www.messenger.com/t/SentinelAI"

                    ret = bot.send_button_message(recipient_id, response_text,
                        [
                            {
                            "type": "postback",
                            "title": "No",
                            "payload": "incorrect_guess" + separator + image_path
                            },
                            {
                            "type": "postback",
                            "title": "Yes",
                            "payload": "correct_guess" + separator + image_path
                            }                                            
                        ])
                    
    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error 
    if token_sent == os.environ['VERIFY_TOKEN']:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


#chooses a random message to send to the user
def get_message():
    sample_responses = ["You are stunning!", "We're proud of you.", "Keep on being you!", "We're greatful to know you :)"]
    # return selected item to the user
    return random.choice(sample_responses)

#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

def main():
    #server_name, server_port, flask_debug = __get_flask_server_params__()
    #initialize_app(app, server_name, server_port)
    app.run(host="0.0.0.0", port=5000)
