import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../styles"

Rectangle {
    id: root
    height: 30
    color: Theme.bgTertiary

    // Drag tracking
    property point dragStartPos: Qt.point(0, 0)
    property point dragOffset: Qt.point(0, 0)

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 16
        anchors.rightMargin: 0
        spacing: 0

        // App Title
        Text {
            text: titlebarBridge.title
            color: Qt.rgba(1, 1, 1, 0.7)
            font.pixelSize: 12
            font.weight: Font.DemiBold
            font.family: "Inter, Pretendard, sans-serif"
        }

        Item { Layout.fillWidth: true }

        // Window Control Buttons
        Row {
            spacing: 0

            // Minimize Button
            Rectangle {
                id: minimizeBtn
                width: 46
                height: 30
                color: minimizeArea.containsMouse ? Qt.rgba(1, 1, 1, 0.06) : "transparent"

                Behavior on color {
                    ColorAnimation { duration: 100 }
                }

                // Horizontal line icon
                Rectangle {
                    anchors.centerIn: parent
                    width: 10
                    height: 1
                    color: Qt.rgba(1, 1, 1, 0.78)
                }

                MouseArea {
                    id: minimizeArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: titlebarBridge.minimize()
                }
            }

            // Maximize Button
            Rectangle {
                id: maximizeBtn
                width: 46
                height: 30
                color: maximizeArea.containsMouse ? Qt.rgba(1, 1, 1, 0.06) : "transparent"

                Behavior on color {
                    ColorAnimation { duration: 100 }
                }

                // Square icon (changes based on maximized state)
                Item {
                    anchors.centerIn: parent
                    width: 10
                    height: 10

                    // Normal state: single square
                    Rectangle {
                        visible: !titlebarBridge.isMaximized
                        anchors.fill: parent
                        color: "transparent"
                        border.width: 1
                        border.color: Qt.rgba(1, 1, 1, 0.78)
                    }

                    // Maximized state: overlapping squares
                    Item {
                        visible: titlebarBridge.isMaximized
                        anchors.fill: parent

                        Rectangle {
                            x: 2
                            y: 0
                            width: 7
                            height: 7
                            color: "transparent"
                            border.width: 1
                            border.color: Qt.rgba(1, 1, 1, 0.78)
                        }

                        Rectangle {
                            x: 0
                            y: 2
                            width: 7
                            height: 7
                            color: Theme.bgTertiary
                            border.width: 1
                            border.color: Qt.rgba(1, 1, 1, 0.78)
                        }
                    }
                }

                MouseArea {
                    id: maximizeArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: titlebarBridge.toggleMaximize()
                }
            }

            // Close Button
            Rectangle {
                id: closeBtn
                width: 46
                height: 30
                color: closeArea.containsMouse ? Theme.danger : "transparent"

                Behavior on color {
                    ColorAnimation { duration: 100 }
                }

                // X icon using Canvas
                Canvas {
                    anchors.centerIn: parent
                    width: 10
                    height: 10

                    property bool isHovered: closeArea.containsMouse

                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.clearRect(0, 0, width, height)
                        ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.78)"
                        ctx.lineWidth = 1
                        ctx.lineCap = "round"
                        ctx.beginPath()
                        ctx.moveTo(1, 1)
                        ctx.lineTo(9, 9)
                        ctx.moveTo(9, 1)
                        ctx.lineTo(1, 9)
                        ctx.stroke()
                    }

                    onIsHoveredChanged: requestPaint()
                }

                MouseArea {
                    id: closeArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: titlebarBridge.close()
                }
            }
        }
    }

    // Drag Area (excluding buttons)
    MouseArea {
        id: dragArea
        anchors.fill: parent
        anchors.rightMargin: 138  // 3 buttons * 46px

        property point clickPos

        onPressed: function(mouse) {
            clickPos = Qt.point(mouse.x, mouse.y)
            root.dragStartPos = Qt.point(mouse.x, mouse.y)
            titlebarBridge.startDrag(mouse.x, mouse.y)
        }

        onPositionChanged: function(mouse) {
            if (pressed) {
                var dx = mouse.x - root.dragStartPos.x
                var dy = mouse.y - root.dragStartPos.y
                root.dragOffset = Qt.point(root.dragOffset.x + dx, root.dragOffset.y + dy)
                titlebarBridge.moveDrag(root.dragOffset.x, root.dragOffset.y)
            }
        }

        onReleased: {
            root.dragOffset = Qt.point(0, 0)
            titlebarBridge.endDrag()
        }

        onDoubleClicked: titlebarBridge.toggleMaximize()
    }
}
