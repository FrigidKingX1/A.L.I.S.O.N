import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    function fmtUptime(s) {
        s = Math.max(0, Math.floor(s || 0))
        var h = Math.floor(s / 3600)
        var m = Math.floor((s % 3600) / 60)
        var sec = s % 60
        function pad(n) { return (n < 10 ? "0" : "") + n }
        return h + ":" + pad(m) + ":" + pad(sec)
    }

    Rectangle {
        anchors.fill: parent
        color: THEME.bg
    }

    Flickable {
        anchors.fill: parent
        anchors.margins: 14
        contentHeight: settingsCol.height
        clip: true

        Column {
            id: settingsCol
            width: parent.width
            spacing: 12

            // ---------------- CORE STATUS ----------------
            Rectangle {
                width: parent.width; height: coreCol.height + 28
                color: THEME.bgSoft; radius: 8; border.color: "#1d2735"
                Column {
                    id: coreCol
                    anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
                    anchors.margins: 14
                    spacing: 8
                    Text {
                        text: "CORE STATUS"
                        color: THEME.cyan; font.pixelSize: 13; font.bold: true
                        font.family: "Consolas, monospace"
                    }
                    Row {
                        spacing: 8
                        Rectangle {
                            width: 10; height: 10; radius: 5
                            anchors.verticalCenter: parent.verticalCenter
                            color: bridge.coreOnline ? "#39d98a" : "#5a6b82"
                        }
                        Text {
                            text: bridge.coreOnline ? "CORE ONLINE" : "core offline"
                            color: bridge.coreOnline ? "#39d98a" : THEME.muted
                            font.pixelSize: 12; font.family: "Consolas, monospace"
                        }
                    }
                    Text {
                        text: "PID: " + (bridge.diagnostics.corePid !== null ? bridge.diagnostics.corePid : "—")
                        color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Text {
                        text: "Uptime: " + fmtUptime(bridge.diagnostics.uptime_s)
                        color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Text {
                        text: "Relaunch attempts: " + bridge.diagnostics.coreFailures
                        color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Text {
                        text: "Core version: " + (bridge.diagnostics.coreVersion || "—")
                        color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Row {
                        spacing: 8
                        Button {
                            text: "Launch Core"
                            width: 130; height: 30
                            background: Rectangle { color: "#10241a"; radius: 5; border.color: "#39d98a" }
                            contentItem: Text {
                                text: parent.text; color: "#39d98a"
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                font.family: "Consolas, monospace"; font.pixelSize: 11
                            }
                            onClicked: bridge.launchCore()
                        }
                        Button {
                            text: "Stop Core"
                            width: 120; height: 30
                            background: Rectangle { color: "#2a1010"; radius: 5; border.color: "#e05656" }
                            contentItem: Text {
                                text: parent.text; color: "#e05656"
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                font.family: "Consolas, monospace"; font.pixelSize: 11
                            }
                            onClicked: bridge.stopCore()
                        }
                        Button {
                            text: "Kill Switch"
                            width: 120; height: 30
                            background: Rectangle { color: "#2a1010"; radius: 5; border.color: "#ff5b5b" }
                            contentItem: Text {
                                text: parent.text; color: "#ff5b5b"
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                font.family: "Consolas, monospace"; font.pixelSize: 11
                            }
                            onClicked: bridge.sendCommand("kill_switch")
                        }
                        Button {
                            text: "Refresh"
                            width: 100; height: 30
                            background: Rectangle { color: "#13202e"; radius: 5; border.color: THEME.cyan }
                            contentItem: Text {
                                text: parent.text; color: THEME.cyan
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                font.family: "Consolas, monospace"; font.pixelSize: 11
                            }
                            onClicked: { bridge.refreshDiagnostics(); bridge.fetchStatus() }
                        }
                    }
                }
            }

            // ---------------- PATHS & MODEL ----------------
            Rectangle {
                width: parent.width; height: pathsCol.height + 28
                color: THEME.bgSoft; radius: 8; border.color: "#1d2735"
                Column {
                    id: pathsCol
                    anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
                    anchors.margins: 14
                    spacing: 7
                    Text {
                        text: "PATHS & MODEL"
                        color: THEME.cyan; font.pixelSize: 13; font.bold: true
                        font.family: "Consolas, monospace"
                    }
                    Text { text: "Install: " + bridge.diagnostics.installDir; color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"; wrapMode: Text.WrapAnywhere; width: parent.width }
                    Text { text: "State: " + bridge.diagnostics.stateDir; color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"; wrapMode: Text.WrapAnywhere; width: parent.width }
                    Text { text: "Models: " + bridge.diagnostics.modelsDir; color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"; wrapMode: Text.WrapAnywhere; width: parent.width }
                    Text { text: "Model file: " + bridge.diagnostics.modelFile; color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"; wrapMode: Text.WrapAnywhere; width: parent.width }
                    Text {
                        text: "Model size: " + bridge.diagnostics.modelSizeGB.toFixed(2) + " GB   (loaded: " +
                              (bridge.diagnostics.modelLoaded ? "yes" : "no") + ")"
                        color: bridge.diagnostics.modelLoaded ? "#39d98a" : THEME.text
                        font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Row {
                        spacing: 8
                        Text {
                            text: "Log: " + bridge.diagnostics.logPath
                            color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace"
                            elide: Text.ElideMiddle; width: 560
                            anchors.verticalCenter: parent.verticalCenter
                        }
                        Button {
                            text: "Open Log"
                            width: 110; height: 26
                            background: Rectangle { color: "#13202e"; radius: 5; border.color: THEME.cyan }
                            contentItem: Text {
                                text: parent.text; color: THEME.cyan
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                font.family: "Consolas, monospace"; font.pixelSize: 11
                            }
                            onClicked: bridge.openLog()
                        }
                    }
                }
            }

            // ---------------- RUNTIME ----------------
            Rectangle {
                width: parent.width; height: runCol.height + 28
                color: THEME.bgSoft; radius: 8; border.color: "#1d2735"
                Column {
                    id: runCol
                    anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
                    anchors.margins: 14
                    spacing: 7
                    Text {
                        text: "RUNTIME"
                        color: THEME.cyan; font.pixelSize: 13; font.bold: true
                        font.family: "Consolas, monospace"
                    }
                    Text { text: "Device: " + (bridge.diagnostics.device || "—"); color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace" }
                    Text { text: "GPU: " + (bridge.diagnostics.gpuName || "—"); color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace" }
                    Text { text: "RAM: " + bridge.diagnostics.ramGB.toFixed(1) + " GB"; color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace" }
                    Text { text: "VRAM: " + (bridge.diagnostics.vramGB > 0 ? bridge.diagnostics.vramGB.toFixed(1) + " GB" : "—"); color: THEME.text; font.pixelSize: 11; font.family: "Consolas, monospace" }
                    Text { text: "Precision γ: " + (bridge.diagnostics.gamma !== null ? bridge.diagnostics.gamma.toFixed(3) : "—"); color: THEME.violet; font.pixelSize: 11; font.family: "Consolas, monospace" }
                    Text {
                        text: "Screen sense: " + (bridge.diagnostics.screenSense ? "ON" : "off")
                        color: bridge.diagnostics.screenSense ? "#39d98a" : THEME.muted
                        font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Text {
                        text: "Screenpipe ingest: " + (bridge.diagnostics.screenpipe ? "ON" : "off")
                        color: bridge.diagnostics.screenpipe ? "#39d98a" : THEME.muted
                        font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                    Text {
                        text: "Action executor: " + (bridge.diagnostics.actionExecutor === null ? "unknown" :
                            (bridge.diagnostics.actionExecutor ? "enabled" : "disabled"))
                        color: bridge.diagnostics.actionExecutor === true ? "#39d98a" :
                               bridge.diagnostics.actionExecutor === false ? "#e05656" : THEME.muted
                        font.pixelSize: 11; font.family: "Consolas, monospace"
                    }
                }
            }

            // ---------------- LOG TAIL ----------------
            Rectangle {
                width: parent.width; height: 240
                color: THEME.bgSoft; radius: 8; border.color: "#1d2735"
                Column {
                    anchors.fill: parent; anchors.margins: 14
                    spacing: 8
                    Text {
                        text: "CORE LOG (TAIL)"
                        color: THEME.cyan; font.pixelSize: 13; font.bold: true
                        font.family: "Consolas, monospace"
                    }
                    Rectangle {
                        width: parent.width; height: 168
                        color: "#080b11"; radius: 6; border.color: "#1d2735"
                        Flickable {
                            anchors.fill: parent; anchors.margins: 8
                            contentHeight: logText.height
                            clip: true
                            Text {
                                id: logText
                                text: bridge.diagnostics.logTail.length > 0 ? bridge.diagnostics.logTail : "(log empty)"
                                color: THEME.text
                                font.pixelSize: 10; font.family: "Consolas, monospace"
                                wrapMode: Text.WrapAnywhere
                                width: parent.width
                            }
                        }
                    }
                }
            }
        }
    }
}