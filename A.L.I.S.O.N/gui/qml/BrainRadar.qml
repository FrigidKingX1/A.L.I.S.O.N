import QtQuick 2.15

Item {
    id: root
    property real timeVal: 0

    Timer {
        interval: 16; running: true; repeat: true
        onTriggered: timeVal += 0.016
    }

    ShaderEffect {
        id: radar
        anchors.fill: parent
        fragmentShader: "shaders/brain_radar.frag"
        property real u_gamma: bridge.gamma
        property var u_drives: bridge.drives
        property real u_time: timeVal
    }

    // Drive labels around the radar
    Repeater {
        model: THEME.driveLabels
        Text {
            property real a: (index * 60) * Math.PI / 180.0
            x: root.width / 2 + Math.cos(a) * (root.width * 0.42) - width / 2
            y: root.height / 2 - Math.sin(a) * (root.height * 0.42) - height / 2
            text: modelData
            color: THEME.muted
            font.pixelSize: 11
            font.family: "Consolas, monospace"
        }
    }
}
