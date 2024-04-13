# livekit-agents

Example agents I've built using the LiveKit Agents (https://github.com/livekit/agents) framework

## To run any of these agents

1. `cd` into agent subdirectory
2. `cp .env.example .env`
3. open the `.env` file and replace `XXXXXX` with proper values for each environment variable
4. `python agent.py start`

## To interact with the agent in the agents playground

1. open a browser and navigate to: `https://agents-playground.livekit.io/`
2. choose to the same LiveKit Cloud project that the agent is running against (or manually enter a websocket URL and participant token if self-hosting)
3. click `Connect` in the top right corner of the playground UI
