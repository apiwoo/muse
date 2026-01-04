import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../styles"

Rectangle {
    id: root

    property string title: ""
    property bool collapsible: false
    property bool collapsed: false
    default property alias content: contentColumn.data

    implicitHeight: collapsed ? titleHeight : titleHeight + contentColumn.implicitHeight + Theme.spacingMedium * 2
    implicitWidth: 280

    readonly property int titleHeight: 28

    color: Qt.rgba(1, 1, 1, 0.02)
    border.color: Qt.rgba(1, 1, 1, 0.04)
    border.width: 1
    radius: Theme.radiusLarge

    Behavior on implicitHeight {
        NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutQuad }
    }

    // Content container
    ColumnLayout {
        id: contentColumn
        anchors.fill: parent
        anchors.margins: Theme.spacingMedium
        anchors.topMargin: root.titleHeight
        spacing: Theme.spacingMedium
        visible: !root.collapsed
        opacity: root.collapsed ? 0 : 1

        Behavior on opacity {
            NumberAnimation { duration: Theme.animFast }
        }
    }

    // Title label (floating on top-left border)
    Rectangle {
        x: 14
        y: -8
        color: Theme.bgPrimary
        width: titleRow.width + 12
        height: titleRow.height
        visible: root.title !== ""

        Row {
            id: titleRow
            anchors.centerIn: parent
            spacing: 6

            Text {
                id: titleText
                text: root.title
                color: Theme.accentCyan
                font.pixelSize: Theme.fontSizeSmall
                font.weight: Font.DemiBold
                font.letterSpacing: 1
                textFormat: Text.PlainText
            }

            // Collapse indicator
            Text {
                visible: root.collapsible
                text: root.collapsed ? "+" : "-"
                color: Theme.textSecondary
                font.pixelSize: Theme.fontSizeSmall
                font.weight: Font.Bold
            }
        }

        MouseArea {
            anchors.fill: parent
            enabled: root.collapsible
            cursorShape: root.collapsible ? Qt.PointingHandCursor : Qt.ArrowCursor
            onClicked: root.collapsed = !root.collapsed
        }
    }

    // Subtle glow effect on hover
    Rectangle {
        anchors.fill: parent
        radius: parent.radius
        color: "transparent"
        border.color: Theme.accentCyan
        border.width: 1
        opacity: hoverArea.containsMouse ? 0.15 : 0

        Behavior on opacity {
            NumberAnimation { duration: Theme.animFast }
        }
    }

    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
    }
}
