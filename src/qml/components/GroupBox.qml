import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../styles"

Item {
    id: root

    property string title: ""
    property bool collapsible: false
    property bool collapsed: false
    default property alias content: contentColumn.data

    implicitHeight: collapsed ? titleHeight : titleHeight + Theme.spacingSmall + contentColumn.implicitHeight
    implicitWidth: 280

    readonly property int titleHeight: root.title !== "" ? 18 : 0

    Behavior on implicitHeight {
        NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutQuad }
    }

    // Section Title (Discord style: uppercase, small, muted)
    Item {
        id: titleContainer
        anchors.top: parent.top
        anchors.left: parent.left
        width: titleRow.width
        height: titleRow.height
        visible: root.title !== ""

        Row {
            id: titleRow
            spacing: 6

            Text {
                id: titleText
                text: root.title.toUpperCase()
                color: Theme.textMuted
                font.pixelSize: Theme.fontSizeXSmall
                font.weight: Font.Bold
                font.letterSpacing: 0.5
            }

            // Collapse indicator
            Text {
                visible: root.collapsible
                text: root.collapsed ? "+" : "-"
                color: Theme.textFaint
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

    // Divider line under title
    Rectangle {
        id: divider
        anchors.top: titleContainer.bottom
        anchors.topMargin: Theme.spacingXSmall
        anchors.left: parent.left
        anchors.right: parent.right
        height: 1
        color: Theme.borderSubtle
        visible: root.title !== ""
    }

    // Content
    ColumnLayout {
        id: contentColumn
        anchors.top: root.title !== "" ? divider.bottom : parent.top
        anchors.topMargin: root.title !== "" ? Theme.spacingMedium : 0
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: Theme.spacingSmall
        visible: !root.collapsed
        opacity: root.collapsed ? 0 : 1

        Behavior on opacity {
            NumberAnimation { duration: Theme.animFast }
        }
    }
}
