import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root
    width: 1100
    height: 720

    Rectangle {
        anchors.fill: parent
        color: THEME.bg
    }

    // Title bar
    Rectangle {
        id: title
        width: parent.width; height: 54
        color: "#0b0f16"
        border.color: "#16202c"
        Row {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left; anchors.leftMargin: 18
            spacing: 12
            Text {
                text: THEME.appName
                color: THEME.cyan
                font.pixelSize: 22; font.bold: true
                font.family: "Consolas, monospace"
            }
            Text {
                text: "Adaptive Learning Interface for Sentient Operating Networks"
                color: THEME.muted
                font.pixelSize: 11
                font.family: "Consolas, monospace"
                anchors.verticalCenter: parent.verticalCenter
            }
        }
        Text {
            anchors.right: parent.right; anchors.rightMargin: 18
            anchors.verticalCenter: parent.verticalCenter
            text: "γ " + bridge.gamma.toFixed(2)
            color: THEME.violet
            font.pixelSize: 14; font.family: "Consolas, monospace"
        }
    }

    // Left: brain radar
    Rectangle {
        x: 18; y: 70; width: 520; height: 520
        color: "#080b11"; radius: 10; border.color: "#16202c"
        BrainRadar { anchors.fill: parent; anchors.margins: 14 }
    }

    // Right: control panel + buttons
    ControlPanel {
        x: 560; y: 70; width: 320
    }

    Button {
        x: 560; y: 350; width: 320; height: 34
        text: "Toggle Ambient Overlay (Alt+Space)"
        background: Rectangle { color: "#101a26"; radius: 6; border.color: THEME.cyan }
        contentItem: Text {
            text: parent.text; color: THEME.cyan; horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter; font.family: "Consolas, monospace"
        }
        onClicked: bridge.toggleOverlay()
    }

    // Bottom-left: hippocampal trace
    HippocampalView {
        x: 18; y: 604; width: 520; height: 100
    }

    // Bottom-right: event log
    Rectangle {
        x: 560; y: 396; width: 522; height: 308
        color: THEME.bgSoft; radius: 8; border.color: "#1d2735"
        Text {
            text: "EVENT STREAM"
            anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 8
            color: THEME.muted; font.pixelSize: 10; font.family: "Consolas, monospace"
        }
        ListView {
            id: logView
            anchors.fill: parent; anchors.margins: 26
            model: ListModel { id: logModel }
            delegate: Text {
                text: model.text
                color: model.color
                font.pixelSize: 11; font.family: "Consolas, monospace"
                elide: Text.ElideRight; width: logView.width
            }
            clip: true
        }
    }

    Connections {
        target: bridge
        onEventReceived: function (topic, text) {
            var c = THEME.text
            if (topic === "token") c = THEME.cyan
            else if (topic === "thought") c = THEME.violet
            else if (topic === "control") c = "#39d98a"
            else if (topic === "screen") c = THEME.muted
            logModel.append({ text: "[" + topic + "] " + text, color: c })
            if (logModel.count > 200) logModel.remove(0)
            logView.positionViewAtEnd()
        }
    }
}
