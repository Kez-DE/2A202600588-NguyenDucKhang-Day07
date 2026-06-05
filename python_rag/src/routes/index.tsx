import { createFileRoute } from "@tanstack/react-router";
import RagApp from "@/components/RagApp";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Python Docs RAG Agent" },
      { name: "description", content: "Trợ lý truy xuất tài liệu Python với RAG." },
    ],
  }),
  component: Index,
});

function Index() {
  return <RagApp />;
}
