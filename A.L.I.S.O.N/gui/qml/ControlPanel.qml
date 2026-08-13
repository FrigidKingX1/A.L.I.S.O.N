import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root
    implicitWidth: 320
    implicitHeight: 260

    Rectangle {
        anchors.fill: parent
        color: THEME.bgSoft
        radius: 8
        border.color: "#1d2735"
    }

    Column {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 10

        Text {
            text: "CONTROL"
            color: THEME.cyan
            font.pixelSize: 13
            font.family: "Consolas, monospace"
            font.bold: true
        }

        // Gamma (precision) bounds
        Text {
            text: "Precision (γ) bounds"
            color: THEME.text
            font.pixelSize: 11
        }
        Row {
            spacing: 8
            Slider {
                id: lowS
                from: 0.0; to: 4.0; value: 1.0; width: 110; height: 22
            }
            Slider {
                id: highS
                from: 0.0; to: 4.0; value: 3.0; width: 110; height: 22
            }
        }
        Button {
            text: "Apply γ bounds"
            width: 230; height: 30
            background: Rectangle { color: "#13202e"; radius: 5; border.color: THEME.cyan }
            contentItem: Text {
                text: parent.text; color: THEME.cyan; horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter; font.family: "Consolas, monospace"
            }
            onClicked: bridge.sendCommand("set_gamma_bounds",
                { "low": lowS.value, "high": highS.value })
        }

        Row { spacing: 8
            Button {
                text: "Screen Sense"
                width: 150; height: 30
                background: Rectangle { color: "#13202e"; radius: 5; border.color: THEME.violet }
                contentItem: Text {
                    text: parent.text; color: THEME.violet; horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter; font.family: "Consolas, monospace"
                }
                onClicked: bridge.sendCommand("toggle_screen_sense")
            }
            Button {
                text: "Status"
                width: 72; height: 30
                background: Rectangle { color: "#13202e"; radius: 5; border.color: THEME.cyan }
                contentItem: Text {
                    text: parent.text; color: THEME.cyan; horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter; font.family: "Consolas, monospace"
                }
                onClicked: bridge.sendCommand("get_status")
            }
        }

        Button {
            text: "Launch Core"
            width: 230; height: 30
            background: Rectangle { color: "#10241a"; radius: 5; border.color: "#39d98a" }
            contentItem: Text {
                text: parent.text; color: "#39d98a"; horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter; font.family: "Consolas, monospace"
            }
            onClicked: bridge.launchCore()
        }

        Text {
            text: "status: " + (bridge.coreOnline ? "CORE ONLINE" : "core offline")
            color: bridge.coreOnline ? "#39d98a" : THEME.muted
            font.pixelSize: 11
            font.family: "Consolas, monospace"
        }
    }
}
