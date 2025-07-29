import logfire

logfire.configure(service_name="vi-faq-bot")
print("💡 logfire INITIALISED from observation.py", flush=True) 

logger = logfire          
span   = logfire.span    
