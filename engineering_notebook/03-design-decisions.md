# 3. Record Design Decisions

Log design choices with supporting code snippets, diagrams, and rationale. Use this page to explain why certain approaches were chosen.

## Voice Agent as Discord Voice (discord.js-selfbot-v13)

### Summary
To make the voice agent the voice in Discord using `discord.js-selfbot-v13`, follow these steps:

1. **Install the library**:
   ```sh
   npm install discord.js-selfbot-v13@latest
   ```
2. **Authenticate**: Use your Discord user token (note: this violates Discord ToS and risks account ban).
3. **Join a Voice Channel**:
   ```js
   const { Client } = require('discord.js-selfbot-v13');
   const client = new Client();
   client.on('ready', () => {
     const channel = client.channels.cache.get('VOICE_CHANNEL_ID');
     if (channel && channel.join) {
       channel.join().then(connection => {
         // Play audio or stream TTS here
       });
     }
   });
   client.login('your_token_here');
   ```
4. **Stream Audio**: Use the connection to play TTS or other audio.

### Warnings
- This approach is against Discord's Terms of Service and may result in account suspension.
- Use only for research or non-production accounts.

### Next Steps
- Integrate your TTS system to generate audio and stream it to the Discord voice connection.
- Handle connection lifecycle and error management.
