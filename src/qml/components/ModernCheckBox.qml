import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../styles"

Item {
    id: root
    implicitHeight: 36
    implicitWidth: 200

    property string text: "Option"
    property string description: ""
    property bool checked: false

    signal toggled(bool newValue)

    RowLayout {
        anchors.fill: parent
        spacing: Theme.spacingMedium

        // Label and description
        Column {
            Layout.fillWidth: true
            spacing: 2

            Text {
                text: root.text
                color: Theme.textNormal
                font.pixelSize: Theme.fontSizeMedium
                font.weight: Font.Normal
            }

            Text {
                visible: root.description !== ""
                text: root.description
                color: Theme.textMuted
                font.pixelSize: Theme.fontSizeSmall
                width: parent.width
                wrapMode: Text.WordWrap
            }
        }

        // Toggle Switch (Discord style)
        Rectangle {
            id: toggle
            width: Theme.toggleWidth
            height: Theme.toggleHeight
            radius: height / 2
            color: root.checked ? Theme.green : Theme.textFaint

            Behavior on color {
                ColorAnimation { duration: Theme.animNormal }
            }

            // Handle
            Rectangle {
                id: handle
                width: 18
                height: 18
                radius: 9
                color: "#ffffff"
                y: (parent.height - height) / 2
                x: root.checked ? parent.width - width - 3 : 3

                Behavior on x {
                    NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutQuad }
                }
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    root.checked = !root.checked
                    root.toggled(root.checked)
                }
            }
        }
    }
}
