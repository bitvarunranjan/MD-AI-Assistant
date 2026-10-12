import os, json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title='MD Assistant v0.5 - Synthetic Demo')
origins = [s.strip() for s in os.getenv('ALLOWED_ORIGINS','').split(',') if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['POST','GET'], allow_headers=['Content-Type'])

SYSTEM = '''You are an administrative decision-support assistant for SYNTHETIC training cases only. Analyze the supplied case strictly as hypothetical. Never invent rules, delegations, orders, dates, or citations. Distinguish facts asserted in text from documentary verification. A document title mentioned in a case is not independently verified. Do not conclude illegality without applicable rule evidence. Be proportionate and avoid unnecessary objections. Return valid JSON with keys proposal (string), established (array of strings), concerns (array of objects with issue, importance, basis, action), missing_evidence (array of strings), recommendation (string), draft_noting (string). If no concerns, return empty concerns. Cite exact excerpts of the provided text in basis; do not invent page numbers. Draft noting should be brief, conditional, and human-reviewable.'''

class ExamineRequest(BaseModel):
    text: str = Field(min_length=30, max_length=30000)

@app.get('/health')
def health(): return {'status':'ok','mode':'synthetic-only'}

@app.post('/examine')
def examine(payload: ExamineRequest):
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(503, 'Server model key is not configured')
    try:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=35)
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'),
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'Examine this SYNTHETIC case:\n'+payload.text}],
            response_format={'type':'json_object'}, temperature=0.1,
        )
        result = json.loads(response.choices[0].message.content)
        return {'analysis':result,'disclaimer':'Synthetic training use only. AI output is not verified legal or administrative advice.'}
       except Exception as exc:
        import logging
        logging.getLogger("uvicorn.error").exception(
            "AI examination failed"
        )
        raise HTTPException(
            status_code=502,
            detail="Model examination failed. Check server logs."
        ) from exc

