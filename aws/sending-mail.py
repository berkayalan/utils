import boto3
def publish_to_sns(sub, msg):
    topic_arn = ""
    sns = boto3.client("sns")
    response = sns.publish(
        TopicArn=topic_arn,
        Message=msg,
        Subject=sub
    )

def send_email():
    sub = ""
    msg = """

        """
    publish_to_sns(sub, msg)

send_email()