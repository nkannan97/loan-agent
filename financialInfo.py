from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from plaid.api import plaid_api
from plaid.model import *
from plaid import Configuration, ApiClient
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Plaid Configuration
configuration = Configuration(
    host=PlaidEnvironments[os.getenv("PLAID_ENV")],
    api_key={
        "clientId": os.getenv("PLAID_CLIENT_ID"),
        "secret": os.getenv("PLAID_SECRET"),
    }
)
api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Input schema
class TokenExchange(BaseModel):
    public_token: str

@app.post("/exchange_token")
def exchange_token(data: TokenExchange):
    try:
        exchange_request = ItemPublicTokenExchangeRequest(public_token=data.public_token)
        exchange_response = client.item_public_token_exchange(exchange_request)
        return {"access_token": exchange_response.access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accounts")
def get_accounts(access_token: str):
    try:
        request = AccountsGetRequest(access_token=access_token)
        response = client.accounts_get(request)
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/identity")
def get_identity(access_token: str):
    try:
        request = IdentityGetRequest(access_token=access_token)
        response = client.identity_get(request)
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/income")
def get_income(access_token: str):
    try:
        request = IncomeGetRequest(access_token=access_token)
        response = client.income_get(request)
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
def get_transactions(access_token: str, start_date: str, end_date: str):
    try:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start_date,
            end_date=end_date
        )
        response = client.transactions_get(request)
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
