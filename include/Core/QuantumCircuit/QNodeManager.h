/*! \file QNodeManager.h */
#ifndef _MANAGER_H
#define _MANAGER_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/Tools/ReadWriteLock.h"

QPANDA_BEGIN

class QNodeManager
{
public:
	QNodeManager(const QNode* parent);
	QNodeManager() = delete;
	~QNodeManager();

	void push_back_node(std::shared_ptr<QNode> node);
	NodeIter get_first_node_iter();
	NodeIter get_last_node_iter();
	NodeIter get_end_node_iter();
	NodeIter get_head_node_iter();
	NodeIter insert_QNode(const NodeIter &perIter, std::shared_ptr<QNode> node);
	NodeIter delete_QNode(NodeIter &target_iter);
	void clear();

protected:
	QNodeManager(const QNodeManager&);
	QNodeManager& operator=(const QNodeManager&);

private:
	const QNode* m_parent;
	Item *m_head{ nullptr };
	Item *m_end{ nullptr };
	SharedMutex m_sm;
};

QPANDA_END

#endif // !_MANAGER_H