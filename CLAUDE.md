# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project workspace containing several independent applications. The main projects are:

1. **TaskOwl** - Task management application (React, Express, MongoDB)
2. **01 Project** - Open-source AI voice assistant hardware/software
3. **Not a Label** - Platform for independent musicians

## Common Commands

### TaskOwl (Root Project)
```bash
npm install          # Install dependencies
npm test            # Run tests
```

### 01 Project
```bash
cd 01
poetry install      # Install dependencies
poetry run 01       # Run the application
poetry run 01 --server --expose  # Run in server mode with external access
```

### Not a Label
```bash
cd "Not a Label"
./setup-dev-environment.sh  # Initial setup
./start-dev.sh             # Start development servers
```

## Architecture Overview

### Technology Stack
- **Frontend**: React, Next.js, TypeScript
- **Backend**: Node.js/Express, Python (Poetry)
- **Databases**: MongoDB (Mongoose), PostgreSQL, Supabase
- **AI/ML**: OpenAI API integration, Llama models
- **Authentication**: JWT with bcrypt
- **Deployment**: Vercel, Docker

### 01 Project Architecture
- Speech-to-speech websocket server on localhost:10001
- LMC (Language Model Computer) message protocol
- ESP32-based hardware integration for voice interface
- Supports both local and server modes

### Not a Label Architecture
- Separate frontend/backend repositories
- Progressive Web App with offline capabilities
- AI-powered career assistant for musicians
- Analytics dashboard with data visualization

## Security Notes
Several files contain exposed API keys that need to be secured. Always use environment variables for sensitive credentials.