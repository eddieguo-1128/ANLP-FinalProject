
"""Example Python client for vllm.entrypoints.api_server"""
# https://github.com/vllm-project/vllm/blob/main/examples/api_client.py
import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:1528/v1/"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]
prompt = "{'intent': 'Show me japanese restaurants in Fresno with online delivery. on yellowpages', 'obs': \"\\n\\\\t[1272] textbox \\\\'\\\\' required: False\\n\\\\t[1311] textbox \\\\'\\\\' required: False\\n\\\\t[910] button \\\\'Find\\\\'\\n\\\\t[913] StaticText \\\\'QUESTIONS & ANSWERS\\\\'\\n\\\\t[911] StaticText \\\\'CITY PAGES\\\\'\\n\\\\t[1144] StaticText \\\\'POPULAR CITIES\\\\'\\n\\\\t[1145] link \\\\'View All Cities \u00c2\u00bb\\\\'\\n\\\\t[1147] link \\\\'Albuquerque\\\\'\\n\\\\t[1148] link \\\\'Atlanta\\\\'\\n\\\\t[1149] link \\\\'Austin\\\\'\\n\\\\t[1150] link \\\\'Bakersfield\\\\'\\n\\\\t[1151] link \\\\'Baltimore\\\\'\\n\\\\t[1152] link \\\\'Baton Rouge\\\\'\\n\\\\t[1153] link \\\\'Birmingham\\\\'\\n\\\\t[1154] link \\\\'Bronx\\\\'\\n\\\\t[1155] link \\\\'Brooklyn\\\\'\\n\\\\t[1156] link \\\\'Charlotte\\\\'\\n\\\\t[1158] link \\\\'Chicago\\\\'\\n\\\\t[1159] link \\\\'Cincinnati\\\\'\\n\\\\t[1160] link \\\\'Cleveland\\\\'\\n\\\\t[1161] link \\\\'Columbia\\\\'\\n\\\\t[1162] link \\\\'Columbus\\\\'\\n\\\\t[1163] link \\\\'Corpus Christi\\\\'\\n\\\\t[1164] link \\\\'Dallas\\\\'\\n\\\\t[1165] link \\\\'Denver\\\\'\\n\\\\t[1166] link \\\\'Detroit\\\\'\\n\\\\t[1167] link \\\\'El Paso\\\\'\\n\\\\t[1169] link \\\\'Fort Lauderdale\\\\'\\n\\\\t[1170] link \\\\'Fort Worth\\\\'\\n\\\\t[1171] link \\\\'Fresno\\\\'\\n\\\\t[1172] link \\\\'Houston\\\\'\\n\\\\t[1173] link \\\\'Indianapolis\\\\'\\n\\\\t[1174] link \\\\'Jacksonville\\\\'\\n\\\\t[1175] link \\\\'Kansas City\\\\'\\n\\\\t[1176] link \\\\'Knoxville\\\\'\\n\\\\t[1177] link \\\\'Las Vegas\\\\'\\n\\\\t[1178] link \\\\'Long Island\\\\'\\n\\\\t[1180] link \\\\'Los Angeles\\\\'\\n\\\\t[1181] link \\\\'Louisville\\\\'\\n\\\\t[1182] link \\\\'Memphis\\\\'\\n\\\\t[1183] link \\\\'Miami\\\\'\\n\\\\t[1184] link \\\\'Milwaukee\\\\'\\n\\\\t[1185] link \\\\'Nashville\\\\'\\n\\\\t[1186] link \\\\'New Orleans\\\\'\\n\\\\t[1187] link \\\\'New York\\\\'\\n\\\\t[1188] link \\\\'Oklahoma City\\\\'\\n\\\\t[1189] link \\\\'Orlando\\\\'\\n\\\\t[1191] link \\\\'Philadelphia\\\\'\\n\\\\t[1192] link \\\\'Phoenix\\\\'\\n\\\\t[1193] link \\\\'Sacramento\\\\'\\n\\\\t[1194] link \\\\'Saint Louis\\\\'\\n\\\\t[1195] link \\\\'Salt Lake City\\\\'\\n\\\\t[1196] link \\\\'San Antonio\\\\'\\n\\\\t[1197] link \\\\'San Diego\\\\'\\n\\\\t[1198] link \\\\'Tampa\\\\'\\n\\\\t[1199] link \\\\'Tucson\\\\'\\n\\\\t[1200] link \\\\'Tulsa\\\\'\\n\\\\t[864] link \\\\'WRITE A REVIEW\\\\'\\n\\\\t[865] link \\\\'ADVERTISE WITH US\\\\'\\n\\\\t[868] link \\\\'The Real Yellow Pages logo\\\\'\\n\\\\t\\\\t[1006] img \\\\'The Real Yellow Pages logo\\\\'\\n\\\\t[1007] StaticText \\\\'Discover Local\u00e2\u201e\\\\\\\\xa0\\\\'\\n\\\\t[1008] LineBreak \\\\'\\\\\\n\\\\'\\n\\\\t[1009] StaticText \\\\'Local businesses need your support. Spend where it matters.\\\\'\\n\\\\t[1203] link \\\\'Find People\\\\'\\n\\\\t[1204] link \\\\'Auto Repair\\\\'\\n\\\\t[1211] link \\\\'Veterinarians\\\\'\\n\\\\t\\\\t[1383] StaticText \\\\'JS\\\\'\\n\\\\t\\\\t[1384] StaticText \\\\'Hi James S.\\\\'\\n\\\\t[1221] link \\\\'Coupons\\\\'\\n\\\\t[1019] StaticText \\\\'Update your business information in a few steps.\\\\'\\n\\\\t[1020] LineBreak \\\\'\\\\\\n\\\\'\\n\\\\t[1021] StaticText \\\\'Make it easy for your customers to find you on Yellowpages.\\\\'\\n\\\\t[1025] heading \\\\'Make Every Day Local \u00c2\u00ae\\\\'\\n\\\\t[1233] link \\\\'Learn more \u00c2\u00bb\\\\'\\n\\\\t\\\\t[1234] link \\\\'Android App Available\\\\'\\n\\\\t\\\\t\\\\t[1395] img \\\\'Android App Available\\\\'\\n\\\\t[1238] ListMarker \\\\'\u2022 \\\\'\\n\\\\t[1241] StaticText \\\\'Find answers or offer solutions\\\\'\\n\\\\t[1038] link \\\\'Dentists\\\\'\\n\\\\t[1039] link \\\\'Family Law\\\\'\\n\\\\t[1041] link \\\\'Auto Repair\\\\'\\n\\\\t[1521] StaticText \\\\'$5 Off $25 eGift Card to Krispy Kreme\\\\'\\n\\\\t[1522] StaticText \\\\'$20\\\\'\\n\\\\t\\\\t\\\\t[1053] StaticText \\\\'About\\\\'\\n\\\\t\\\\t[1055] link \\\\'Contact Us\\\\'\\n\\\\t\\\\t[1057] link \\\\'Corporate Blog\\\\'\\n\\\\t\\\\t[1059] link \\\\'Legal\\\\'\\n\\\\t\\\\t[1061] link \\\\'Terms of Use\\\\'\\n\\\\t\\\\t[1062] link \\\\'Advertising Choices\\\\'\\n\\\\t\\\\t[1074] link \\\\'Find a Business\\\\'\\n\\\\t\\\\t[1076] link \\\\'YP Mobile App\\\\'\\n\\\\t\\\\t[1081] link \\\\'Browse Restaurants\\\\'\\n\\\\t\\\\t[890] HeaderAsNonLandmark \\\\'\\\\'\\n\\\\t\\\\t[1098] link \\\\'Charlotte\\\\'\\n\\\\t\\\\t[1101] link \\\\'Denver\\\\'\\n\\\\t\\\\t[1103] link \\\\'Detroit\\\\'\\n\\\\t\\\\t[1108] link \\\\'Los Angeles\\\\'\\n\\\\t\\\\t[1109] link \\\\'Louisville\\\\'\\n\\\\t\\\\t[1110] link \\\\'Memphis\\\\'\\n\\\\t\\\\t[1116] link \\\\'Orlando\\\\'\\n\\\\t\\\\t[1118] link \\\\'Phoenix\\\\'\\n\\\\t\\\\t[1119] link \\\\'Saint Louis\\\\'\\n\\\\t\\\\t[892] HeaderAsNonLandmark \\\\'\\\\'\\n\\\\t\\\\t[1123] link \\\\'Marketing Solutions\\\\'\\n\\\\t\\\\t[1124] link \\\\'AnyWho\\\\'\\n\\\\t\\\\t[896] link \\\\'Follow us on Twitter\\\\'\\n\\\\t\\\\t[900] link \\\\'Do Not Sell My Personal Information\\\\'\\n\\\\t\\\\t[905] StaticText \\\\'YP, the YP logo and all other YP marks contained herein are trademarks of YP LLC and/or YP affiliated companies.\\\\'\\n\\\\t\\\\t[906] StaticText \\\\'All other marks contained herein are the property of their respective owners.\\\\'', 'image': array([[[238, 238, 238, 255],\\n        [238, 238, 238, 255],\\n        [238, 238, 238, 255],\\n        ...,\\n        [238, 238, 238, 255],\\n        [238, 238, 238, 255],\\n        [238, 238, 238, 255]],\\n\\n       [[229, 229, 229, 255],\\n        [229, 229, 229, 255],\\n        [229, 229, 229, 255],\\n        ...,\\n        [229, 229, 229, 255],\\n        [229, 229, 229, 255],\\n        [229, 229, 229, 255]],\\n\\n       [[255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        ...,\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255]],\\n\\n       ...,\\n\\n       [[255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        ...,\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255]],\\n\\n       [[255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        ...,\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255]],\\n\\n       [[255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        ...,\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255],\\n        [255, 255, 255, 255]]], dtype=uint8)}\", 'action_hist': [\"CLICK [148] where 148 is  StaticText \\\\'CITY PAGES\\\\'\\\\n\\\\t\"]}"
 # Completion API
stream = False
completion = openai.Completion.create(
    model=model,
    prompt=prompt,
    echo=False,
    n=1,
    stream=stream,
    logprobs=5,
    temperature=0.9)

print("Completion results:")
print(completion)
if stream:
    print(completion['choices'][0]['text'])
else:
    print(completion['choices'][0]['text'])

