#include "QNodeManager.h"
#include <stdexcept>

USING_QPANDA

QNodeManager::QNodeManager(const QNode* parent)
	:m_parent(parent)
{
	m_head = new OriginItem();
	m_head->setNext(m_head);
	m_head->setPre(m_head);
	m_end = m_head;
}

QNodeManager::~QNodeManager()
{
	Item *temp;

	while (m_head->getNext() != m_end)
	{
		temp = m_head->getNext();
		m_head->setNext(temp->getNext());
		delete temp;
	}
	delete m_head;
	m_head = nullptr;
	m_end = nullptr;
}

void QNodeManager::push_back_node(std::shared_ptr<QNode> node)
{
	if (!node)
	{
		QCERR("node is null");
		throw std::runtime_error("node is null");
	}

	if (m_parent == node.get())
	{
		throw std::runtime_error("Error: Cann't inserte to node-self.");
	}

	WriteLock wl(m_sm);

	{
		auto last_node = m_end->getPre();

		Item *iter = new OriginItem();
		iter->setNode(node);
		iter->setNext(m_end);
		iter->setPre(last_node);

		last_node->setNext(iter);
		m_end->setPre(iter);
	}
}

NodeIter QNodeManager::get_first_node_iter()
{
	ReadLock rl(m_sm);
	NodeIter temp(m_head->getNext());
	return temp;
}

NodeIter QNodeManager::get_last_node_iter()
{
	ReadLock rl(m_sm);
	NodeIter temp(m_end->getPre());
	return temp;
}

NodeIter QNodeManager::get_end_node_iter()
{
	NodeIter temp(m_end);
	return temp;
}

NodeIter QNodeManager::get_head_node_iter()
{
	NodeIter temp(m_head);
	return temp;
}

NodeIter QNodeManager::insert_QNode(const NodeIter &perIter, std::shared_ptr<QNode> node)
{
	ReadLock * rl = new ReadLock(m_sm);

	if (m_parent == node.get())
	{
		throw std::runtime_error("Error: Cann't inserte to node-self.");
	}

	if (perIter == m_head)
	{
		delete rl;
		WriteLock wl(m_sm);
		Item *new_iter = new OriginItem();
		new_iter->setNode(node);

		auto first_node = m_head->getNext();
		new_iter->setNext(first_node);
		new_iter->setPre(m_head);

		first_node->setPre(new_iter);
		m_head->setNext(new_iter);

		NodeIter temp(new_iter);
		return temp;
	}

	Item * perItem = perIter.getPCur();
	if (nullptr == perItem)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	auto aiter = this->get_first_node_iter();

	for (; aiter != this->get_end_node_iter(); aiter++)
	{
		if (perItem == aiter.getPCur())
		{
			break;
		}
	}
	if (aiter == this->get_end_node_iter())
	{
		QCERR("The perIter is not in the qprog");
		throw std::runtime_error("The perIter is not in the qprog");
	}

	delete rl;
	WriteLock wl(m_sm);
	Item *curItem = new OriginItem();
	curItem->setNode(node);
	if (m_end != perItem->getNext())
	{
		perItem->getNext()->setPre(curItem);
		curItem->setNext(perItem->getNext());
		perItem->setNext(curItem);
		curItem->setPre(perItem);
	}
	else
	{
		auto last_node = m_end->getPre();
		curItem->setNext(m_end);
		curItem->setPre(last_node);
		last_node->setNext(curItem);
		m_end->setPre(curItem);
	}

	NodeIter temp(curItem);
	return temp;
}

NodeIter QNodeManager::delete_QNode(NodeIter &target_iter) 
{
	ReadLock *rl = new ReadLock(m_sm);
	Item * target_item = target_iter.getPCur();
	if (nullptr == target_item)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	auto aiter = this->get_first_node_iter();
	for (; aiter != this->get_end_node_iter(); aiter++)
	{
		if (target_item == aiter.getPCur())
		{
			break;
		}
	}
	if (aiter == this->get_end_node_iter())
	{
		QCERR("The target_iter is not in the qprogget_iter");
		throw std::runtime_error("The target_iter is not in the qprogget_iter");
	}


	delete rl;
	WriteLock wl(m_sm);

	if (m_head == target_item)
	{

		m_head = target_item->getNext();
		m_head->setPre(nullptr);
		delete target_item;
		target_iter.setPCur(nullptr);
		NodeIter temp(m_head);
		return temp;
	}

	Item * perItem = target_item->getPre();
	if (nullptr == perItem)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	Item * nextItem = target_item->getNext();
	if (nullptr == nextItem)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}
	perItem->setNext(nextItem);
	nextItem->setPre(perItem);
	delete target_item;
	target_iter.setPCur(nullptr);
	NodeIter temp(perItem);
	return temp;
}

void QNodeManager::clear()
{
	WriteLock wl(m_sm);
	Item *temp;

	while (m_head->getNext() != m_end)
	{
		temp = m_head->getNext();
		m_head->setNext(temp->getNext());
        temp->getNext()->setPre(m_head);
		delete temp;
	}
}