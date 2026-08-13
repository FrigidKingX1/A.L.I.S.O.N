import QtQuick 2.15

Item {
    id: root
    property real timeVal: 0

    Timer {
        interval: 16; running: true; repeat: true
        onTriggered: timeVal += 0.016
    }

    MouseArea {
        anchors.fill: parent
        property real lastX: 0
        property real lastY: 0
        onPressed: { lastX = mouseX; lastY = mouseY }
        onPositionChanged: {
            if (pressed) {
                bridge.moveOverlay(mouseX - lastX, mouseY - lastY)
                lastX = mouseX; lastY = mouseY
            }
        }
    }

    ShaderEffect {
        id: arc
        anchors.fill: parent
        fragmentShader: "shaders/ambient_visualizer.frag"
        property real u_time: timeVal
        property real u_audioRMS: bridge.listening ? (0.55 + 0.45 * Math.sin(timeVal * 8.0)) : bridge.activity
        property real u_listening: bridge.listening ? 1.0 : 0.0
        property real u_anxiety: bridge.drives[2]
        property real u_curiosity: bridge.drives[3]
    }

    Text {
        anchors.centerIn: parent
        text: THEME.appName
        color: "#eaf6ff"
        font.pixelSize: 18
        font.bold: true
        font.family: "Consolas, monospace"
        opacity: 0.9
    }
    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.verticalCenter; anchors.topMargin: 22
        text: bridge.coreOnline ? "observing" : "awaiting core"
        color: THEME.cyan
        font.pixelSize: 11
        font.family: "Consolas, monospace"
    }

    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.verticalCenter; anchors.topMargin: 40
        text: "● listening"
        color: "#7CFFCB"
        font.pixelSize: 12
        font.family: "Consolas, monospace"
        visible: bridge.listening
        opacity: 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(timeVal * 6.0))
    }
}
