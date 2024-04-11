document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chat-history');
    const userMessageInput = document.getElementById('user-message');
    const sendButton = document.getElementById('send-button');

    sendButton.addEventListener('click', function() {
        sendMessage();
    });

    userMessageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userMessageInput.value.trim();
        if (userMessage) {
            appendMessage('user', userMessage);
            userMessageInput.value = '';
            sendRequest(userMessage);
        }
    }

    function sendRequest(userMessage) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_query: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.response;
            appendMessage('bot', botResponse);
        })
        .catch(error => console.error('Error:', error));
    }

    function appendMessage(sender, message) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);
        messageContainer.textContent = `${sender}: ${message}`;
        chatHistory.appendChild(messageContainer);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
