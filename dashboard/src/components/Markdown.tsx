import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renderiza markdown da resposta do assistente com a tipografia `.prose-brain`. */
export const Markdown = memo(function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-brain text-[0.94rem] text-ink">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
});
