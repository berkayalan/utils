import requests
import json

print('teams_url')

def send_msteams_notif(title, subtitle ):

    url = f"{teams_url}"


    payload = json.dumps({
      "@type": "MessageCard",
      "@context": "http://schema.org/extensions",
      "themeColor": "{3}",
      "summary": "{0}",
      "sections": [
        {
          "activityTitle": title,
          "activitySubtitle": subtitle,
          "markdown": True,
          "potentialAction": [
            {
              "@type": "OpenUri",
              "name": "{4}",
              "targets": [
                {
                  "os": "default",
                  "uri": "{5}"
                }
              ]
            }
          ]
        }
      ]
    })

    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response