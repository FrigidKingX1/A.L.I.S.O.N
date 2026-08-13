import QtQuick 2.15

Item {
    id: root
    implicitWidth: 420
    implicitHeight: 180

    Rectangle {
        anchors.fill: parent
        color: THEME.bgSoft
        radius: 8
        border.color: "#1d2735"
    }

    Canvas {
        id: cv
        anchors.fill: parent
        anchors.margins: 8
        onPaint: {
            var ctx = cv.getContext("2d")
            var w = cv.width, h = cv.height
            ctx.clearRect(0, 0, w, h)
            var g = ctx.createLinearGradient(0, 0, 0, h)
            g.addColorStop(0, "#0c1018"); g.addColorStop(1, "#070a0e")
            ctx.fillStyle = g; ctx.fillRect(0, 0, w, h)

            var n = hist.length
            if (n < 2) return
            var colors = ["#00E5FF", "#3aa0ff", "#7C4DFF", "#b06bff", "#39d98a", "#ffd166"]
            for (var d = 0; d < 6; d++) {
                ctx.beginPath()
                for (var i = 0; i < n; i++) {
                    var x = i / (maxSamples - 1) * w
                    var y = h - hist[i][d] * h
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                }
                ctx.strokeStyle = colors[d]; ctx.globalAlpha = 0.5
                ctx.lineWidth = 1.5; ctx.stroke()
            }
            // gamma (bright) overlaid, normalised /3
            ctx.beginPath()
            for (var i = 0; i < n; i++) {
                var x = i / (maxSamples - 1) * w
                var y = h - Math.min(1.0, hist[i][6] / 3.0) * h
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
            }
            ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.85
            ctx.lineWidth = 1.5; ctx.stroke()
            ctx.globalAlpha = 1.0
        }
    }

    property int maxSamples: 240
    property var hist: []

    function pushSample() {
        var s = [bridge.drives[0], bridge.drives[1], bridge.drives[2],
                 bridge.drives[3], bridge.drives[4], bridge.drives[5], bridge.gamma]
        hist.push(s)
        if (hist.length > maxSamples) hist.shift()
        cv.requestPaint()
    }

    Connections {
        target: bridge
        onTelemetryUpdated: pushSample()
    }

    Text {
        text: "HIPPOCAMPAL TRACE  (6 drives + γ)"
        anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 8
        color: THEME.muted
        font.pixelSize: 10
        font.family: "Consolas, monospace"
    }
}
