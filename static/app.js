// FIXME: Add a user name entry box

// Input Enable/Disable

let inputEnabled = false;

enableInput(false);

function enableInput(enable) {
    const textInput = document.getElementById("textInput");
    const sendButton = document.getElementById("sendButton");
    const nameButton = document.getElementById("nameButton");

    if (enable) {
        sendButton.innerHTML = "Submit";
        sendButton.className = "submit";

        nameButton.innerHTML = "Change";
    } else {
        textInput.value = "";

        sendButton.innerHTML = "Cancel";
        sendButton.className = "cancel";

        nameButton.innerHTML = "Busy";
    }

    textInput.disabled = !enable;
    //sendButton.disabled = enable; - becomes the cancel button
    nameButton.disabled = !enable;

    inputEnabled = enable;

    console.log("Enabling input:", enable);
}

/* User ID */

// Write a unique identifier to local storage if it doesn't exist
if (!localStorage.getItem('uniqueID')) {
    const uniqueID = Math.random().toString(36);
    localStorage.setItem('uniqueID', uniqueID);
}

/* Chat Box */

// The index of the message matches the children of the conversation element
messages = [];

class ChatMessage {
    constructor(id, sender, message) {
        this.id = id;
        this.sender = sender;
        this.message = message;
    }
}

function findChatMessage(id) {
    for (var i = 0; i < messages.length; i++) {
        if (messages[i].id == id) {
            return i;
        }
    }
    return -1;
}

function clearChatMessages() {
    messages = [];
    var conversation = document.getElementById('conversation');
    while (conversation.firstChild) {
        conversation.removeChild(conversation.firstChild);
    }
}

function removeChatMessage(id) {
    index = findChatMessage(id);
    if (index == -1) {
        return;
    }

    var conversation = document.getElementById('conversation');
    child = conversation.children[index];
    conversation.removeChild(child);
    messages.splice(index, 1);
}

function addChatMessage(id, sender, message) {
    var conversation = document.getElementById('conversation');
    var messageElement = document.createElement('div');

    messageElement.className = 'chat-text';
    
    // Message
    var messagePart = document.createElement('p');
    name_tag = '<span class="' + sender.toLowerCase() + '-tag">' + sender + '</span>';
    separator = '<span class="' + sender.toLowerCase() + '-separator">&gt; </span>';
    messageContent = '<span class="' + sender.toLowerCase() + '-message">' + message + '</span>';
    messagePart.innerHTML = name_tag + separator + messageContent;
    messageElement.appendChild(messagePart);

    // Delete button
    var deleteButton = document.createElement('button');
    deleteButton.innerHTML = '-';
    deleteButton.className = 'delete-button';
    deleteButton.onclick = function() {
        socket.emit("linex_message", { id: id });
    };
    messageElement.appendChild(deleteButton);

    auto_scroll = false;
    if (conversation.scrollTop + 10 >= conversation.scrollHeight - conversation.offsetHeight) {
        auto_scroll = true;
    }

    // Insert into messages[] and conversation element
    index = findChatMessage(id);
    if (index == -1) {
        m = new ChatMessage(id, sender, message);
        conversation.appendChild(messageElement);
        messages.push(m);
        auto_scroll = true;
    } else {
        conversation.replaceChild(messageElement, conversation.children[index]);
        m = messages[index];
        if (index == messages.length - 1) {
            auto_scroll = true;
        }
    }
    m.sender = sender;
    m.message = message;

    if (auto_scroll) {
        // Auto scroll to the bottom (scrollHeight may not be updated yet)
        conversation.offsetHeight;
        setTimeout(() => {
            conversation.scrollTop = conversation.scrollHeight;
        }, 100);
    }
}

/* Record Text */

let recording = false;

function setRecording(isRecording) {
    if (isRecording === recording) {
        return;
    }
    recording = isRecording;
    console.log("Recording: " + isRecording);

    const recordingDot = document.getElementById('recordingDot');
    recordingDot.style.display = isRecording ? 'inline-block' : 'none';

    var instructionElement = document.getElementById('talk-instruction');

    if (isRecording) {
        instructionElement.innerText = '[Listening]';
        instructionElement.classList.add('recording');
    } else {
        instructionElement.innerText = rest_text;
        instructionElement.classList.remove('recording');
    }

    socket.emit("record_message", { recording: isRecording });
}

// Function to detect if the user is on a mobile device
function isMobileDevice() {
    return (typeof window.orientation !== "undefined") || (navigator.userAgent.indexOf('IEMobile') !== -1);
};

// Change the instruction text if the user is on a mobile device
if (isMobileDevice()) {
    rest_text = 'Press and hold screen to talk';
} else {
    rest_text = 'Hold space/mouse to talk';
}

document.getElementById('talk-instruction').textContent = rest_text;

/* Record Trigger */

function stopEventPropagationById(elementId) {
    var element = document.getElementById(elementId);

    if (element) {
        ['mousedown', 'mouseup', 'keydown', 'keyup', 'touchstart', 'touchend'].forEach(function(eventName) {
            element.addEventListener(eventName, function(event) {
                event.stopPropagation();
            });
        });
    } else {
        console.error('Element with ID', elementId, 'not found.');
    }
}

let from_key = false;

document.body.onkeydown = function(e) {
    if (e.key === " ") {
        setRecording(true);
        from_key = true;
    }
};

document.onkeyup = function(e) {
    if (e.key === " ") {
        setRecording(false);
    }
};

document.body.onmousedown = function(e) {
    setRecording(true);
    from_key = false;
};

document.onmouseup = function(e) {
    setRecording(false);
};

document.onmousemove = function(e) {
    if (from_key) {
        return;
    }
    if ((e.buttons & 1) === 0) { // 1 indicates the left button
        setRecording(false);
    }
};

document.body.ontouchstart = function(e) {
    setRecording(true);
    from_key = false;
};

document.ontouchend = function(e) {
    setRecording(false);
};

document.onselectionchange = function() {
    var selection = window.getSelection();
    if (selection.rangeCount > 0) {
        setRecording(false);
    }
};

/* Socket.IO */

const socket = io();
const configuration = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
let peerConnection;

// Status functions

function setStatus(status) {
    placeholder = document.getElementById("placeholder");
    placeholder.innerHTML = status;

    console.log("WebRTC status:", status);
}

// Default is shown
function showStatus() {
    placeholder = document.getElementById("placeholder");
    placeholder.style.display = "block";
}

function hideStatus() {
    placeholder = document.getElementById("placeholder");
    placeholder.style.display = "none";
}

// Socket.io event handlers

socket.on("connect", () => {
    setStatus("Connected to signaling server.  Negotiating video...");
    startWebRTC();

    // Send our browser ID that should persist between refreshes
    const id = localStorage.getItem('uniqueID');
    socket.emit("id_message", { id: id });
});

socket.on("disconnect", () => {
    setStatus("Disconnected from signaling server");
    closeWebRTC();
});

socket.on("session_response", async (data) => {
    console.log("Received SDP response:", data);
    const description = new RTCSessionDescription(data);
    await peerConnection.setRemoteDescription(description);
    setStatus("Set remote description");
});

socket.on("candidate_response", async (data) => {
    console.log("Received ICE candidate response:", data);
    const candidate = new RTCIceCandidate(data.candidate);
    await peerConnection.addIceCandidate(candidate);
    setStatus("Add ICE candidate");
});

socket.on("clear_chat_message", async (data) => {
    console.log("Received clear_chat message:", data);
    clearChatMessages();
});
socket.on("remove_chat_message", async (data) => {
    console.log("Received remove_chat message:", data);
    removeChatMessage(data.id);
});
socket.on("add_chat_message", async (data) => {
    console.log("Received add_chat message:", data);
    addChatMessage(data.id, data.sender, data.message);
});

socket.on("allow_message", async (data) => {
    console.log("Received allow message:", data);
    enableInput(true);
});

// Video functions

async function getMediaStream() {
    const constraints = {
        audio: true,
        video: {
            width: { ideal: 640, max: 1920 },
            height: { ideal: 480, max: 1080 },
        },
    };
    try {
        return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
        setStatus("Error getting media stream:", error);
        return null;
    }
}

// WebRTC functions

async function startWebRTC() {
    try {
        peerConnection = new RTCPeerConnection(configuration);
        peerConnection.onicecandidate = onIceCandidate;
        peerConnection.onconnectionstatechange = onConnectionStateChange;
        peerConnection.ontrack = onTrack;

        // Add media stream to the peer connection
        const stream = await getMediaStream();
        if (stream) {
            stream.getTracks().forEach((track) => peerConnection.addTrack(track, stream));

            const video = document.getElementById("video");
            video.srcObject = stream;
        } else {
            setStatus("Failed to get media stream");
            closeWebRTC();
            return;
        }

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        await socket.emit("session_message", { sdp: offer });
    } catch (error) {
        setStatus("Error starting WebRTC:", error);
        closeWebRTC();
    }
}

async function closeWebRTC() {
    showStatus();
    if (peerConnection) {
        await peerConnection.close();
        peerConnection = null;
    }
}

// WebRTC Event handlers

async function onIceCandidate(event) {
    if (event.candidate) {
        socket.emit("candidate_message", { candidate: event.candidate });
    }
}

async function onConnectionStateChange(event) {
    setStatus(peerConnection.connectionState);

    if (peerConnection.connectionState === "connected") {
        hideStatus();
    } else {
        showStatus();
    }

    if (peerConnection.connectionState === "disconnected" ||
        peerConnection.connectionState === "failed") {
        setStatus("WebRTC disconnected. Reconnecting...");
        await closeWebRTC();
        await startWebRTC();
    }
}

async function onTrack(event) {
    const video = document.getElementById("video");
    const audio = document.getElementById("audio");
    if (event.track.kind === 'video' && !video.srcObject && event.streams[0]) {
        console.log("Received video track!");
        video.srcObject = event.streams[0];
    }
    else if (event.track.kind === 'audio' && !audio.srcObject && event.streams[0]) {
        console.log("Received audio track!");
        audio.srcObject = event.streams[0];
    }
}

// Send Button

function onSendButtonClick() {
    if (!inputEnabled) {
        socket.emit("cancel_message", {});
        return;
    }

    // Grab text input
    const textInput = document.getElementById("textInput");
    const text = textInput.value;

    if (text) {
        enableInput(false);

        socket.emit("chat_message", { message: text });
    }
}

document.getElementById("sendButton").addEventListener("click", onSendButtonClick);

// Name Entry

document.getElementById("user-name").addEventListener("change", function() {
    let userName = document.getElementById("user-name").value;
    console.log("User's name is now " + userName);
});

document.getElementById("ai-name").addEventListener("change", function() {
    let aiName = document.getElementById("ai-name").value;
    console.log("AI's name is now " + aiName);
});

document.getElementById("use-vision").addEventListener("change", function() {
    let useVision = document.getElementById("use-vision").checked;
    socket.emit("use_vision", { value: useVision });
});

document.getElementById("nameButton").addEventListener("click", function() {
    let userName = document.getElementById("user-name").value;
    let aiName = document.getElementById("ai-name").value;
    let useVision = document.getElementById("use-vision").checked;
    console.log("Submitted: User's name is " + userName + ", AI's name is " + aiName);
    if (userName.length > 0) {
        socket.emit("user_name_message", { name: userName });
    }
    if (aiName.length > 0) {
        socket.emit("ai_name_message", { name: aiName });
    }
    socket.emit("use_vision", { value: useVision });
});

/* Elements that should not trigger recording */
stopEventPropagationById('conversation');
stopEventPropagationById('textInput');
stopEventPropagationById('sendButton');
stopEventPropagationById('social-links');
stopEventPropagationById('nameButton');
stopEventPropagationById('user-name');
stopEventPropagationById('ai-name');
stopEventPropagationById('use-vision');
stopEventPropagationById('name-entry-container');
