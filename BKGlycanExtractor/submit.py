import sys
import time
import json
import requests

def searchGlycoCT(seq):

    if type(seq)!=list:
        seq=[seq]
    main_url = "https://edwardslab.bmcb.georgetown.edu/glylookup/"
    # main_url = "http://localhost:10980/"



    params = {
        "q": json.dumps(seq),
    }


    try:
        response1 = requests.post(main_url + "submit", params)
        response_json = response1.json()
        list_ids = list(response_json.values())
    except Exception as e:
        sys.stdout.write("Error: has issue connecting to flask API.")
        sys.stdout.write(str(e))
        sys.exit()

    # the list id for retrieval later
    #print (list_ids,"list_ids here")
    #print ("\n" * 3)

    params = {"q": json.dumps(list_ids)}

    # might require larger wait time in case you send a huge amount of GlycoCTs
    while(1):

        break
    time.sleep(0.3)

    response2 = requests.post(main_url+ "retrieve", params)
    results = response2.json()
    #print("results here",results)

    for list_id,res in results.items():
        #print(f"list_id:{list_id}")
        #print(res['result']['error'])
        #print("res:", res["result"]['hits'])
        if res['finished']==True and res["result"]['hits']!=[]:
            accession=res['result']['hits'][0]
        else:
            return "not found"
        #print(f"hits:{res['result']['hits'][0]}")
        #print(f"entire resposne:{res['result']}")
        #for k, v in res.items():
        #    print(f"k:{k} v:{v}")
        #print ("- " * 20)

    return accession


