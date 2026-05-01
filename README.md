# 🚗 Driving Chatbot (Jordan)

## Overview
This is an AI-powered chatbot that helps students prepare for driving exams in Jordan.  
It answers questions about traffic rules, road signs, and driving safety.

---

## Features
- AI chatbot for driving questions  
- Supports Arabic and English answers  
- Simple and modern UI  
- Real-time chat experience  
- Built with OpenAI API  

---

## Tech Stack
- Next.js  
- React  
- TypeScript  
- Tailwind CSS  
- OpenAI API  

---

## How it works
User → Frontend → API Route → OpenAI → Response → UI  

---

## Project Structure
- `app/page.tsx` → Frontend UI  
- `app/api/chat/route.ts` → Backend API  
- `public/` → Static files  

---

## AI Model
Uses GPT-4o-mini from OpenAI to generate responses based on driving rules.

---

## Technical Details
(finished AI integration backend and frontend → ready for OpenAI connection)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev