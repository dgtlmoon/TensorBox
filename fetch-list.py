import json,sys

for obj in json.load(sys.stdin):
  print "curl %s > data/tshirtslayer/%s" %( obj["original_url"],  obj["image_path"])

