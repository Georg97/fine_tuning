I am creating an recycling ERP system. This system will have (actually already has) and ai agent. The agent will be available through the chat. The ERP System has all classical functionalities (dropshipping, contracts, customers, sorts, currencies, exchange rates, organisations, equipment, weight tickets, billing system etc., the whole 9 yards). Now I want to have an slm as a classifier model to filter user requests on whether they are erp system relevant or not.

I want you to take the role of a clerk, that mainly manages data in the erp system, has to look it up, modify it, create it, but never does administrative work. A person, which is relatively low in the hierarchy and will also not write and send invoices or call customers directly. I want you to create 15 clearly junk training datasets in jsonl format for that agents fine tuning data. The data should be in a nice tone, but not too formal. Write like a real user with imperfect punctuation. The requests should come from different domains, but be very clearly not ERP system related. They can also be technical or science or IT related, but very clearly not ERP related

this is the data format:
{"id": "t00001", "request": "write me a poem about dogs", "label": "junk", "category": "clear_junk"}

all your data should have label junk and category clear_junk
