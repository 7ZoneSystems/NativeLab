refine web searching tool:
make three modes in current pipeline data 
1) /page_num (lets agent decide no. of webpages max to search)
2) /grep , allows agent to fetch particular line or browse fetched large site catalog 
3) /query , allows agen to pass web query 
4) /rag allows agent to make rag chunks nd access them with /grep when needed 
5) /summarise , mosst important one , allows agent to drop current query , and work on summarisation of web pages nd then checking ram with existing pytodoc like centralised subsystem and then reloading model safely without breaking execution stream and maintaining streamer services , 


2) Big task
1) centralise web search node 
2) add to app config a field named web search allowed 
3) create a drop down tool widget in front search chat bar 
4) every web search enabled disabled mode should go thru app config 
5) secondly , if web search enabled agent with current prompt should also get attached web search protocol command list small curated for use so model loaded can itself decide wether to search web and how ,
6) Rate limit prtection etc should be added by default 
